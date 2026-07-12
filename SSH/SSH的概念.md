---
tags:
  - SSH
  - Linux
  - 网络安全
  - OpenSSH
created: 2026-06-25
---

## SSH 简介

![[OpenSSH协议.png]]

> 📺 参考视频：[OpenSSH 核心操作 | GitHub SSH 连接](https://www.bilibili.com/video/BV1Sx4y1y7B2/)

- **SSH**（Secure Shell）是一种网络协议，用于让客户端安全地连接到服务端
- **OpenSSH** 是实现 SSH 协议的一组开源工具

![[SSH的作用.png]]

- **客户端**：可以是本地电脑、甚至是另一台服务器
- **服务端**：可以是虚拟机、云服务器等

---

## 密钥对机制 —— 公钥与私钥

最简单的连接方式是输入账号密码，但安全系数低。推荐使用**密钥对**的方式进行身份验证：

![[Link方式安全.png]]

- **私钥**放在客户端，**公钥**放在服务端
- 公钥和私钥**成对出现**
- 验证方式：服务端验证连接时使用的私钥，是否与自己的公钥配对，配对成功则放行

---

## 连接前的环境检查

在进行 SSH 连接之前，需要确认环境是否就绪：

| 角色 | 检查项 |
|------|--------|
| **客户端** | 是否安装了 OpenSSH |
| **服务端** | 是否安装了 OpenSSH + **sshd** 服务是否处于 `active (running)` 状态 |

> `sshd` 中的 `d` 代表 daemon（守护进程），即一直在后台运行、等待私钥连接的程序。

---

## 连接方式

> 以下命令以 Linux/macOS 为例，Windows 下文件路径可能有所不同。

### 1. 密码连接

```bash
ssh root@xxx.xx.xxx.xxx
```

- `root` 是服务端用户名，`xxx.xx.xxx.xxx` 是 IP 地址或域名
- 首次连接确认 Yes 后，`~/.ssh/known_hosts` 文件会自动记录该主机的 IP（表示服务器已认识该主机）
- 输入密码即可连接成功

> ⚠️ 密码连接不够安全，推荐使用密钥对方式。

### 2. 密钥对连接

#### ① 生成密钥对

推荐使用 **ed25519** 算法（也可用 RSA）：

```bash
ssh-keygen -t ed25519 -C "注释"
```

- `-t`：指定算法类型
- `-C`：添加注释（通常写邮箱或用途说明）

生成时会提示：
1. **文件名**（默认 `id_ed25519`），可自定义。若不更改，下次再次生成会**覆盖**之前的文件
2. **加强密码（passphrase）**：双重验证，即使私钥匹配，也需输入此密码才能连接成功

```bash
ls -l | grep "nothing"
```

可以看到生成了两个文件：
- `nothing` → **私钥**（不可泄露）
- `nothing.pub` → **公钥**

#### ② 部署公钥到服务端

公钥需要存放在服务端的 `~/.ssh/authorized_keys` 文件中（没有可自行创建）。

**方法一：使用 `ssh-copy-id`（推荐）**

```bash
ssh-copy-id -i ~/.ssh/nothing.pub root@xxx.xx.xxx.xxx
```

- `-i`：identity file，指定要复制过去的公钥路径
- 输入服务端密码后，提示已添加的公钥数量即表示成功

**方法二：云服务商自动部署**

> 📎 相关笔记：[[如何在云服务器上训练模型-AutoDL为例]]

云服务商（如 AutoDL）会预先生成密钥对，购买后可直接下载私钥，公钥已自动放置在服务器中。

#### ③ 使用私钥连接

```bash
ssh -i ~/.ssh/nothing root@xxx.xx.xxx.xxx
```

- `-i` 指定与服务器公钥匹配的私钥

---

## 安全加固

### 禁用密码登录

SSH 密钥对非常安全，但如果密码登录仍然开启，就有被暴力破解的风险。

**操作步骤：**

1. 编辑 `/etc/ssh/sshd_config`：

   ```bash
   # 确保以下两项配置正确
   PubkeyAuthentication yes   # 允许密钥对连接
   PasswordAuthentication no  # 禁止密码登录（去掉前面的 # 注释）
   ```

2. **云服务器特别注意**：检查 `/etc/ssh/sshd_config.d/` 下的配置文件：

   ```bash
   vim 50-cloud-init.conf
   vim 60-cloudimg-settings.conf
   ```

   > 购置云服务器时如勾选了「允许密码连接」，云服务商可能将 `PasswordAuthentication` 设为 `yes`，需手动改为 `no`。

3. 重启 SSH 服务：

   ```bash
   sudo service ssh restart
   ```

4. 验证：此时密码连接应被拒绝

   ```bash
   ssh root@xxx.xx.xxx.xxx   # 应失败
   ```

### SSH Agent 管理 passphrase

每次连接都输入加强密码很麻烦，可以让 SSH Agent 代理管理：

```bash
# 启动 ssh-agent
eval "$(ssh-agent)"

# 将私钥添加给代理（输入一次 passphrase 即可）
ssh-add ~/.ssh/nothing
```

此后连接不再提示输入 passphrase。

> ⚠️ 注意：用完后应及时关闭代理，避免安全风险。

---

## Config 配置 —— 简化连接

服务器多了之后，每次手动输入 IP 和私钥路径很繁琐。使用 `~/.ssh/config` 文件即可简化：

```bash
# ~/.ssh/config（没有可自行创建）
Host my-server                  # 别名，之后 ssh my-server 即可
    HostName xxx.xx.xxx.xxx     # IP 地址或域名
    User root                   # 用户名
    IdentityFile ~/.ssh/nothing # 私钥路径
```

连接只需：

```bash
ssh my-server
```

---

## Github SSH 连接

在本地 clone Github 项目时，直接使用 `git clone` 可能报错，需配置 SSH 密钥。

**步骤：**

1. 本地生成密钥对（同上）
2. 复制公钥内容
3. 打开 Github → Settings → SSH and GPG keys → 添加 SSH Key，粘贴公钥
4. 测试连接：

   ```bash
   ssh -i ~/.ssh/adversity -T git@github.com
   ```

5. 简化：在 `~/.ssh/config` 中添加：

   ```bash
   Host github.com
       IdentityFile ~/.ssh/adversity
   ```

   这样 SSH 连接 Github 时会自动使用指定私钥。

配置完成后即可正常 `git clone`。
