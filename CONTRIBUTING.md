# 贡献指南

欢迎提交问题反馈、文档改进和代码修复。请先阅读 [README](README.md)，了解项目的个人部署方式和当前支持范围。

## 本地检查

使用 Python 3.10 或更新版本，在虚拟环境中安装依赖：

```bash
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -v
python bot.py --help
python chatgpt.py --help
python weixin.py --help
```

自动化测试使用模拟模型响应和临时目录，不需要填写 QQ 凭据、登录 ChatGPT 或配置 DeepSeek Key。新增行为应覆盖正常路径、权限边界和相关失败情况；文档修改检查命令、链接与配置是否和代码一致。

真实集成检查需要使用自己的账户和 QQ 机器人。`python bot.py --test-model` 只检查模型连接；端到端验证再启动机器人，从私聊或已绑定群发送文字。网页搜索可单独运行 `python chatgpt.py test --search "联网查一下 Python 官网目前推荐的稳定版"`。

## 提交问题和修改

问题反馈请包含操作系统、Python 版本、所选服务、复现步骤、预期行为和实际结果。日志只保留与问题相关的错误类型和必要片段，移除账户标识、聊天正文、凭据、绑定口令和服务器地址。

Pull Request 请说明解决的问题、主要行为变化和验证结果。中文文档保持 UTF-8，优先用直接陈述；公开示例只填写占位值。涉及用户可见行为时，同步更新 README 或对应文档。

## 代码结构

| 路径 | 职责 |
| --- | --- |
| `bot.py` | QQ 网关、消息路由、绑定、会话设置、SQLite 记忆与回复格式 |
| `chatgpt.py` | ChatGPT 登录、令牌校验与续期、模型列表、Responses 流和网页引用 |
| `weixin.py` | ClawBot 扫码登录、多账号私聊、持久化消息队列与权限过滤 |
| `attachments.py` | QQ/微信附件解析、引用选择、下载解密及 Responses 多模态输入 |
| `test_*.py` | 登录、消息处理、权限、持久化、格式与搜索回归测试 |
| `config.example.json` | 可公开的默认配置 |
| `deploy/qqbotgpt.service` | Linux 常驻服务模板 |

项目使用 [0BSD 许可证](LICENSE)。提交贡献时，请确认有权以同一许可证提供相关代码和文档。
