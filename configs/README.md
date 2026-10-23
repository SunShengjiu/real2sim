# Configurations

`scenes/demo.yaml` 是主流程的场景模板，`robots/piper.yaml` 和 `robots/franka.yaml` 声明固定机器人模型、姿态和当前配准设置。文件路径均相对于各自 YAML 解析。

运行 `real2sim check --config configs/scenes/demo.yaml` 检查输入。先补齐视频与环境，再执行 `real2sim run`。示例模板不代表已经完成真实模型验收。

配置只声明实现支持的参数；未知键直接报错。服务地址和凭据保存在本机 `.env`，不提交。机器人姿态或资产变更需要对应算法/模板支持及重新生成来源记录。
