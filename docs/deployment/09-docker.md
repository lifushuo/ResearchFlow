# Deploy with Docker

Everything runs in containers. The only host port used is the frontend's.

## Artifacts

| Artifact | Path | Contents |
|---|---|---|
| API image | `backend/Dockerfile`, target `api` | Python slim, non-root, migrate-on-start, uvicorn |
| Worker image | `backend/Dockerfile`, target `worker` | Same base, Celery entrypoint |
| Frontend image | `frontend/Dockerfile` | Next.js standalone output |
| Dev infrastructure | `docker-compose.yml` | Postgres and Redis only; the app runs natively |
| Full stack | `docker-compose.full.yml` | api, worker, frontend, postgres, redis |

Both images are multi-stage, run as a non-root user, pin their base image, declare a
`HEALTHCHECK`, and bake in no secrets.

**The frontend image builds from the repository root, not from `frontend/`.** The docs site
renders the `docs/` tree at build time, and `docs/` sits outside a `./frontend` context.
Compose and the release workflow both set `context: .` for this reason.

## The full stack

```bash
./start.sh
```

Or directly:

```bash
cp .env.example .env
# Set JWT_SECRET_KEY and a provider key in .env
docker compose -f docker-compose.full.yml up --build
```

Five services come up:

| Service | Image | Published |
|---|---|---|
| `postgres` | `pgvector/pgvector:pg16` | No |
| `redis` | `redis:7-alpine` | No |
| `api` | built from `backend/`, target `api` | No |
| `worker` | built from `backend/`, target `worker` | No |
| `frontend` | built from the repo root | **Yes** — `${FRONTEND_PORT:-3031}` |

**Postgres must be a pgvector image.** Migration 0006 enables the extension and 0007 creates
a vector column, so a stock image fails `alembic upgrade head` outright.

`extra_hosts: host.docker.internal:host-gateway` is declared on `api` and `worker`, so
`OLLAMA_BASE_URL=http://host.docker.internal:11434/v1` reaches an Ollama on the host —
portably, including on Linux.

Volumes are named distinctly from the dev compose file's, so a dev database of a different
Postgres major version is not mounted into the full stack. This does **not** isolate container
identity: using both files with the same Compose project replaces services named `postgres`
and `redis`. Do not run a bare `docker compose up` from an existing full-stack installation.

### Windows PowerShell：包含评测中心的完整服务

已有 `.env` 时，在仓库根目录运行：

```powershell
.\start-evaluation.ps1 -Build
```

该脚本明确使用完整服务与评测扩展两个配置，固定项目 `researchflow`，检查原命名卷及 PostgreSQL 16，
构建当前代码后等待服务就绪。仅恢复已有镜像去掉 `-Build`；`-CheckOnly` 只检查配置，不启动服务。
脚本不重写 `.env`，不删除容器数据卷、不清理历史评测，也不主动创建模型评测任务。
默认包含在线评测 Worker；`-OfflineOnly` 不启动该可选 profile，也不会停止已在运行的在线 Worker。

等价的明确配置命令为：

```powershell
docker compose -p researchflow -f docker-compose.full.yml -f docker-compose.evaluation.override.yml --profile eval-online up -d --build
```

裸 `docker compose up -d --build` 读取的是仅有开发数据库/Redis 的 `docker-compose.yml`，
可能替换已有完整服务容器，并在宿主机 5432/6379 发生端口冲突。完整服务的数据卷是
`mara_full_postgres_data`、`mara_full_redis_data`、`mara_full_corpus_data`，评测卷是
`researchflow_eval_job_data`；不要用新创建的 `researchflow_postgres_data` 空卷代替原库。
误启动时先停止写入、备份并核对原卷版本，再用上述配置恢复；不要使用 `down -v` 或卷清理。

**A locally built image reports the commit it was built from, or `unknown`.** Settings →
About and `GET /api/v1/version` read a build stamp. `./start.sh` reads the version, commit
and whether the tree had uncommitted changes from git, and the `api` and `worker` images
record them. A plain `docker compose … up --build`, or a `docker build` of `backend/` by
hand, has no git to read, so those images answer `unknown`. No local build reuses a stamp
left in the checkout by an earlier one: the backend Dockerfile discards it unless told
otherwise, and only the release workflow, which stamps the images itself, tells it otherwise.
The same holds running from source: if the checkout contradicts a stamp it holds, the API
answers `unknown`.

**A fourth named volume, `mara_full_corpus_data`, holds per-project corpora** (docs/12
M10) and is mounted at `/app/data` on **both** `api` and `worker`. It must be on both:
they are separate containers with separate writable layers, so a document uploaded
through `api` and not visible to `worker` is exactly the split-brain a shared volume
exists to prevent. Without this volume at all, a corpus lives only in a container's
writable layer and is destroyed by any `up --build` or `down` recreate — `./start.sh
--stop` / a plain `docker compose down` preserves it, and only `./start.sh --reset` /
`down -v` removes it, same as the database.

To back up a project's corpus outside the volume:

```bash
docker run --rm -v mara_full_corpus_data:/data -v "$PWD":/backup alpine \
  tar czf /backup/corpus-backup.tar.gz -C /data .
```

## Startup ordering

Migrations run exactly once per deploy, and dependants wait for them:

1. `postgres` and `redis` come up and pass their own healthchecks.
2. `api` starts, runs `alembic upgrade head` in its entrypoint, then serves.
3. `api`'s healthcheck polls `/health/ready`, which checks the **database and Redis are
   actually reachable** rather than merely that the process is alive.
4. `worker` and `frontend` declare `depends_on: api: {condition: service_healthy}`, so
   neither starts against an unmigrated schema.

This is why readiness is a separate endpoint from liveness. Gating on "the process is up"
would start the worker against a database that has not been migrated yet.

## Development infrastructure only

For working on the code, run the datastores in Docker and the app natively:

```bash
make infra-up          # postgres + redis
make infra-down
make infra-clean       # also removes volumes
```

Details in the [Development guide](../developers/32-development.md).

## Configuration

All configuration is environment variables, read from `./.env`. Secrets are never baked into
images.

Compose overrides `DATABASE_URL` and `REDIS_URL` with the internal service hostnames, so the
values in `.env` only matter for a native run.

See [Configuration](../getting-started/21-configuration.md) and the
[configuration reference](../reference/36-configuration.md).

## Rebuilding after a change

The frontend container is a **static build**, not a bind-mounted dev server. Source edits
need a rebuild; a browser reload shows you stale UI:

```bash
docker compose -f docker-compose.full.yml build frontend
docker compose -f docker-compose.full.yml up -d frontend
```

## Migrations

- The API entrypoint runs `alembic upgrade head` before starting uvicorn.
- Rolling back one revision is `alembic downgrade -1`, and the forward/backward round-trip
  is exercised in CI.
- **No `create_all` in application code** — Alembic owns the schema. The LangGraph
  checkpoint tables are created by the library's own setup, invoked from a migration, so
  there is one history.

Running a migration by hand:

```bash
docker compose -f docker-compose.full.yml exec api alembic upgrade head
docker compose -f docker-compose.full.yml exec api alembic current
```

## Health checks

| Endpoint | Meaning |
|---|---|
| `GET /health` | Liveness — the process is up. Used by the container healthcheck |
| `GET /health/ready` | Readiness — database and Redis both answered. **503** until they do |

The worker's healthcheck pings the broker and the worker itself, and fails if either is
unreachable.

## Images and releases

Tagging `vX.Y.Z` triggers the release workflow, which builds multi-architecture (`amd64` and
`arm64`) api, worker, and frontend images to GHCR and cuts a GitHub Release. The desktop
workflow builds and attaches the desktop bundles with a `SHA256SUMS` file.

## GitHub Pages in a personal repository

The Pages workflow always builds and checks the static site. In a personal copy, publishing
is opt-in: first select **GitHub Actions** under **Settings → Pages**, then set the repository
Actions variable `ENABLE_PAGES_DEPLOYMENT` to `true`. Re-run the Pages workflow to publish.
The original upstream repository keeps its existing automatic publishing behaviour.

Without this opt-in the build and artifact upload still run, but deployment is skipped.
The workflow does not try to create a Pages site with its restricted `GITHUB_TOKEN`.
This setting only affects the public documentation site, not the self-hosted Docker app.

Next: [Production deployment](30-production.md) · [Operations](31-operations.md)
