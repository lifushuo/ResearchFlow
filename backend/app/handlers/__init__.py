"""共用应用层：每项产品操作只有一个标准实现。

这里存放普通 async 函数，通过参数接收数据库、用户身份和端口。禁止依赖 FastAPI、服务器配置、Celery、Redis、系统密钥库或 desktop。
两种主机只包装轻量路由，依赖方向由 test_layer_boundaries 验证。"""
