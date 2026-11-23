注意：关于整个test文件夹，这个暂时没想好，你直接跳过，留一个空文件夹，啥都不用管。后面所有提到的这个test文件夹全部不用管。当没看见。

## 1. 项目背景与重构目标

目前 GP 数据采集已在 Notebook 中实现并测试了以下数据源：

1.  **arXiv**：论文 ID 筛选、v1 元数据采集。
2.  **DBLP**：作者身份识别及作者历史特征。
3.  **Semantic Scholar**：三年引用数 C3、三年 influential citation 数
    IC3，以及引用关系记录。
4.  **OpenReview**：公开评审评分采集，并聚合为 paper-level mean rating。

当前阶段的工作不是重新设计爬虫或改变研究方法，而是把 Notebook
中已经验证的流程迁移为结构清晰的 Python
项目，并补足统一配置、缓存、断点续跑、错误管理、数据检查和部署能力。

**禁止为了代码简洁而重写或简化已经验证的核心算法。**



## 2. 建议的项目目录

建议采用统一的小写目录名，并将配置、中控、状态管理和测试分开：

``` text
fetch/
├── __init__.py
│
├── arxiv/
│   ├── __init__.py
│   ├── paper_id.py
│   └── paper_metadata.py
│
├── dblp/
│   ├── __init__.py
│   └── author_info.py
│
├── semantic_scholar/
│   ├── __init__.py
│   └── citation.py
│
├── openreview/
│   ├── __init__.py
│   └── review_score.py
│
├── control/
│   ├── __init__.py
│   ├── config.py
│   ├── tools.py
│   ├── state.py
│   ├── fetch.py
│   ├── combine.py
│   └── check.py
│
├── tests/
│   ├── test_config.py
│   ├── test_cache.py
│   ├── test_retry.py
│   └── test_master.py
│
├── requirements.txt
└── note.md
```

### 目录职责

  --------------------------------------------------------------------------------
  文件                                职责
  ----------------------------------- --------------------------------------------
  `arxiv/paper_id.py`                 按年份和筛选条件获取候选 arXiv ID

  `arxiv/paper_metadata.py`           获取 arXiv v1 元数据

  `dblp/author_info.py`               DBLP 作者匹配、身份判断和历史特征

  `semantic_scholar/citation.py`      Semantic Scholar 匹配、C3、IC3 和 citation
                                      edges

  `openreview/review_score.py`        OpenReview 评分抓取及 paper-level 聚合

  `control/config.py`                 所有全局配置的唯一来源

  `control/tools.py`                  HTTP、重试、缓存、文件读写、日志等通用工具

  `control/state.py`                  运行状态、阶段状态、断点和恢复信息

  `control/fetch.py`                  整个采集流程的中控入口

  `control/combine.py`                合并一篇论文一行的 Master Table

  `control/check.py`                  数据覆盖率、质量检查和磁盘占用报告

  `tests/`                            核心机制及回归测试
  --------------------------------------------------------------------------------

所有 Python 包目录均应包含
`__init__.py`。目录命名统一使用小写，避免跨操作系统时出现大小写路径问题。

## 3. 全局配置要求

所有可调整参数必须集中在
`control/config.py`，其他模块不得各自维护互相冲突的配置副本。

建议使用 `dataclass` 或等价的统一配置对象。

### 3.1 必须提供的配置

  配置项                                      默认值 说明
  ------------------------- ------------------------ ------------------------
  `storage_mode`                             `local` 存储模式
  `data_root`                 `./GP_Research_Papers` 项目数据根目录
  `start_year`                                `2017` 起始年份，包含该年
  `end_year`                                  `2022` 终止年份，包含该年
  `target_per_year`                            `100` 每年最多获取的论文数量
  `random_seed`                               `3544` 可复现抽样的随机种子
  `citation_window_years`                        `3` 引用时间窗口，默认三年
  `cache_policy`                       `reuse_valid` 默认缓存策略
  `resume_enabled`                            `true` 是否启用断点续跑
  `max_retries`                                  `7` 单次请求最大重试次数
  `request_timeout`                            `120` HTTP 超时时间
  `wait_for_reconnection`                     `true` 网络中断时是否等待恢复
  `log_level`                                 `INFO` 日志级别

注意：当前默认年份为 2017--2022，首尾年份均包含；每年最多 100
篇意味着六年最多约 600 篇，而不是总共 100 篇。

### 3.2 存储环境

至少支持：

-   `local`：本地电脑或服务器的指定目录。
-   `colab_drive`：Google Colab 挂载的 Google Drive。

存储路径必须由统一配置决定。不得在导入模块时自动执行 Google Drive
挂载；只有当用户选择 `colab_drive` 时才执行相应挂载操作。

未来更换电脑或服务器时，数据采集逻辑不应需要修改。

### 3.3 配置验证

启动时必须验证：

-   `start_year <= end_year`；
-   `target_per_year` 为合法正整数；
-   `citation_window_years` 为合法正数；
-   存储路径可创建、可读写；
-   缓存策略属于支持的选项；
-   运行环境和依赖满足要求。

配置错误应在正式抓取开始前明确报错，不能运行到一半才发现。


## 4. 缓存机制

默认缓存策略应为：

``` python
cache_policy = "reuse_valid"
```

不再使用简单的 `FORCE_REFRESH = True/False` 作为主要缓存控制方式。

### 4.1 缓存策略

  ------------------------------------------------------------------------------------------------
  策略                                行为
  ----------------------------------- ------------------------------------------------------------
  `reuse_valid`                       默认。有效缓存直接复用；缺失、损坏或不匹配的缓存才重新获取

  `refresh_source`                    仅刷新指定数据源

  `refresh_all`                       明确要求时才刷新全部数据
  ------------------------------------------------------------------------------------------------

正式部署默认使用 `reuse_valid`。

### 4.2 缓存分层

建议区分：

  缓存层               内容                               目的
  -------------------- ---------------------------------- ------------------
  Raw Cache            API 原始 JSON、XML、响应页等       避免重复网络下载
  Intermediate Cache   匹配结果、作者索引、分页处理结果   避免重复计算
  Processed Cache      最终特征表、标签表                 避免重复生成

### 4.3 缓存有效性

**不能仅凭文件存在就认定缓存有效。**

缓存至少应记录：

-   数据源名称；
-   请求参数或请求指纹；
-   Schema Version；
-   文件生成时间；
-   数据完整性状态；
-   文件校验信息；
-   必要时记录算法或处理版本。

只有文件存在、可读取、结构正确且请求参数与版本匹配，才能复用。

缓存写入应尽可能采用原子写入方式：先写临时文件，写入成功并校验后再替换正式文件，避免中断时留下半个文件。

### 4.4 保留已有缓存

必须尽量复用目前已经验证的 DBLP V3.1.1 缓存和 Semantic Scholar
缓存。工程化重构不得无理由删除、覆盖或使这些缓存失效。

OpenReview
的公开评分可能发生变化。默认应保留已有成功快照；需要更新时由用户显式选择
`refresh_source`，并记录抓取时间。


## 5. API Key 与请求策略

本项目本次部署要求使用公开接口，不强制用户输入 API Key。

必须做到：

-   不要求用户输入 API Key；
-   不使用 `getpass()` 强制提示；
-   不在代码中硬编码密钥；
-   不把密钥或认证信息写入日志；
-   不因为没有 API Key 就自动判定某个数据源采集失败。

但必须明确：

**不使用 API Key 不代表不会遇到限流。**

不同数据源的服务规则不同，不能让所有数据源共用一个不加区分的高并发配置。应分别控制请求间隔、并发数和重试策略，并遵守各数据源的公开访问规则。

## 6. 网络鲁棒性：429、断网与重试

网络异常不应直接导致整个程序无记录地终止，也不能把失败请求误认为没有数据。

### 6.1 HTTP 429

请求处理建议：

1.  优先遵循响应中的 `Retry-After`。
2.  没有 `Retry-After` 时使用指数退避。
3.  增加随机抖动，减少多个请求同时重试。
4.  对不同数据源分别配置请求频率。
5.  连续遇到 429 时降低请求频率或暂停该数据源。
6.  达到单次重试预算后，持久化失败状态和待重试任务。
7.  不得把请求失败转换为 `NO_MATCH` 或有效的零值。

重试次数耗尽后，可以暂停该来源并继续其他独立来源；后续重新运行时，应能够恢复未完成请求。

### 6.2 断网恢复

建议流程：

1.  检测到连接异常。
2.  暂停当前数据源的网络请求。
3.  保存已成功的 Raw Cache、分页位置、论文状态和 Checkpoint。
4.  按配置等待网络恢复。
5.  网络恢复后，从失败任务继续。
6.  更新恢复时间、失败次数和最终状态。

如果 Colab Runtime 或 Python Kernel
被系统终止，程序无法在后台继续运行。因此，必须把 Checkpoint
和缓存持久化到可靠存储位置。下次重新执行 `fetch.py`
时，再从持久化状态恢复。

### 6.3 失败与无结果必须区分

至少区分以下状态：

-   `COMPLETE`：成功完成；
-   `ZERO_CONFIRMED`：有证据确认结果为零；
-   `NO_MATCH`：请求成功完成，但没有匹配结果；
-   `REQUEST_ERROR`：网络或服务请求失败；
-   `PARTIAL`：只获得部分数据；
-   `PENDING`：尚未处理；
-   `NOT_COLLECTED`：当前版本未采集该数据源。

不能把 `REQUEST_ERROR`、`PARTIAL` 或 `PENDING` 伪装成 `NO_MATCH` 或 0。

## 7. 新增 state.py：运行状态和断点续跑

建议新增：

``` text
fetch/control/state.py
```

该模块负责记录每次运行和每个阶段的真实状态。

示例：

``` json
{
  "run_id": "GP_2026_001",
  "status": "RUNNING",
  "current_step": "semantic_scholar",
  "steps": {
    "arxiv_seed": "SUCCESS",
    "arxiv_metadata": "SUCCESS",
    "dblp": "SUCCESS",
    "semantic_scholar": "RUNNING",
    "openreview": "PENDING",
    "combine": "PENDING",
    "check": "PENDING"
  }
}
```

每完成一个阶段，应及时持久化状态。

如果 Semantic Scholar
处理到一半发生断网，应保留此前已经成功完成的分页和论文结果。恢复运行时，不应重新抓取已经成功且缓存有效的部分。

建议另设 `errors.jsonl` 或等价错误日志，记录：

  字段                说明
  ------------------- ---------------------
  `timestamp`         错误发生时间
  `run_id`            所属运行批次
  `step`              所属模块
  `paper_id`          相关论文 ID，如适用
  `error_type`        错误类别
  `message`           错误信息
  `retry_count`       已尝试次数
  `recovery_status`   是否已恢复

日志中不得写入 API Key、密码或其他敏感认证信息。

# 8. fetch.py：统一中控程序

`control/fetch.py`
是整个项目的唯一主入口，负责协调各个模块，而不是容纳所有数据抓取和科学计算代码。

建议执行顺序：

1.  Load Config：读取并验证全局配置。
2.  Preflight Check：检查依赖、存储目录、磁盘空间和基本网络状态。
3.  arXiv Paper ID Collection：获取目标论文清单。
4.  arXiv v1 Metadata：获取论文 v1 元数据。
5.  DBLP Author Information：作者身份识别和历史特征。
6.  Semantic Scholar Citation：获取 C3、IC3 和 citation edges。
7.  OpenReview Review Score：获取公开评分并生成 paper-level mean。
8.  Combine Master Table：合并为一篇论文一行的主表。
9.  Final Check：执行完整性、覆盖率和磁盘检查。
10. Generate Final Report：输出运行摘要、错误记录和文件清单。

### 8.1 依赖关系与失败处理

不同数据源之间应允许适当独立：

  模块失败            后续处理
  ------------------- ------------------------------------------------------
  arXiv Paper ID      停止整个 Pipeline
  arXiv v1 Metadata   停止依赖该主表的下游流程
  DBLP                记录失败，允许其他独立数据源继续
  Semantic Scholar    记录失败，允许其他独立数据源继续
  OpenReview          记录失败，允许其他独立数据源继续
  Combine             必要输入缺失时，明确标记部分结果或停止，不得假装完整
  Check               必须报告所有未完成项和异常

中控程序应明确区分：

-   整个 Pipeline 是否成功；
-   单个数据源是否成功；
-   最终主表是否完整；
-   是否存在可恢复的失败任务。

不能因为某一个数据源失败，就让所有独立来源都无意义地停止；也不能在某个来源失败时仍把整个项目报告为完全成功。

### 8.2 模块统一返回值

每个数据源模块应返回结构一致的运行摘要，例如：

``` python
{
    "source": "semantic_scholar",
    "status": "SUCCESS",
    "input_papers": 599,
    "processed_papers": 599,
    "failed_papers": 0,
    "cache_hits": 542,
    "network_requests": 12,
    "output_files": [],
    "errors": []
}
```

中控程序负责汇总这些结果，不应通过解析大量 `print()`
文本来判断模块是否成功。

## 9. 必须保留的研究定义与算法

工程化重构不得擅自改变以下定义。

  数据源             必须保留的定义
  ------------------ ------------------------------------------------------
  arXiv              使用 v1 元数据；首次提交日期作为 `t0`
  DBLP               保留 V3.1.1 作者身份匹配、身份冲突处理和时间排除逻辑
  Semantic Scholar   C3 与 IC3 使用相同的三年窗口
  OpenReview         多位评委评分聚合为 paper-level mean rating

### 9.1 C3 与 IC3

必须保持：

\[ C_3 `\geq `{=tex}IC_3 `\geq 0`{=tex} \]

并且：

-   没有引用与引用数据缺失必须区分；
-   零值必须有明确证据；
-   不完整 C3 不得强制填 0；
-   不完整记录不得冒充 `COMPLETE`；
-   C3、IC3 与底层 citation edges 必须保持定义一致。

### 9.2 DBLP

必须保留已验证的 DBLP V3.1.1 核心逻辑，尤其包括：

-   作者身份冲突处理；
-   作者匹配置信度；
-   时间错位处理；
-   已有两阶段缓存机制；
-   不确定的作者身份不得被强行认定为匹配。

### 9.3 OpenReview

必须保留：

-   reviewer-level 原始评分记录；
-   paper-level 平均评分表；
-   没有公开评分的论文可以没有数值；
-   没有评分不等于评分为 0；
-   最终 Master Table 一篇论文一行。

## 10. 数据表粒度与主键约束

各表的观察单位必须明确，不能因作者或评委数量而让论文主表产生重复行。

  表                            一行代表
  ----------------------------- ----------------------------
  arXiv Paper Table             一篇论文
  arXiv Author Table            一位作者在一篇论文中的署名
  DBLP Author Detail            一位目标论文作者的身份记录
  DBLP Paper Features           一篇论文
  Semantic Scholar Metadata     一篇论文
  Citation Edges                一条唯一引用关系
  OpenReview Reviewer Ratings   一条评审评分记录
  OpenReview Paper Ratings      一篇论文
  Final Master Table            一篇论文

最终 Master Table 必须满足：

``` python
master_df["arxiv_id"].is_unique is True
```

并且：

``` python
len(master_df) == papers_arxiv_v1["arxiv_id"].nunique()
```

所有来源合并前必须先确认该来源的数据粒度正确。不能直接将多行作者表、评委表或引用边表合并到论文级主表。

推荐以 arXiv v1
论文表为主表，通过左连接保留所有目标论文。缺失值必须保留为缺失，不能因为某个来源未匹配就删除论文。

## 11. 回归测试：使用现有 599 篇论文作为 Golden Dataset
这个暂时没想好，你直接跳过，留一个空文件夹，啥都不用管。
后面所有提到的这个test文件夹全部不用管。当没看见。

## 12. 代码风格与可维护性

  项目       要求
  ---------- ------------------------------------------------
  注释       每个文件的大步骤使用中文注释
  变量命名   统一使用清晰的英文名称
  空行       保留合理的函数和逻辑分隔，不要过度压缩
  输出       仅输出阶段进度、关键统计和最终结果
  错误       详细写入日志，终端不刷屏
  函数       每个函数只承担一个主要职责
  类型       重要函数使用 Type Hints
  文档       每个模块说明输入、输出、主键、表粒度和文件路径
  依赖       提供明确的 `requirements.txt`
  可复现性   保存配置快照、随机种子、运行 ID 和版本信息

`tools.py` 不应成为所有业务逻辑的集合。它主要负责：

-   HTTP Session；
-   Retry；
-   Rate Limit；
-   网络恢复；
-   Cache；
-   文件读写；
-   日志。

数据源特定的匹配逻辑和科学计算应保留在对应数据源模块中。

## 13. 建议的开发与交付阶段

不要让 AI 一次性重写所有代码。建议分三个阶段实施。

### 阶段一：工程化封装

-   建立项目目录。
-   将 Notebook 代码迁移为模块。
-   建立统一 Config、Tools、State。
-   保持原有算法不变。
-   使用 599 篇历史缓存进行回归测试。
-   检查输出字段和表格粒度。

**阶段一验收：** 使用已有缓存运行时，研究结果与原 Notebook 保持一致。

### 阶段二：鲁棒性建设

-   实现统一 Retry。
-   实现 429 退避和请求频率控制。
-   实现断网等待与恢复。
-   实现持久化 Checkpoint。
-   实现断点续跑。
-   实现错误分类和日志。
-   测试重复运行是否真正复用有效缓存。

**阶段二验收：**
模拟请求失败、429、断网和进程中断后，能够保存状态并恢复；失败不能被伪装为无匹配或零值。

### 阶段三：正式部署

-   Local 环境测试。
-   Colab Drive 环境测试。
-   100 篇小规模测试。
-   599 篇完整回归测试。
-   检查输出文件和磁盘占用。
-   确认日志、状态和错误报告。
-   通过验收后再扩大到正式采集规模。

## 14. 最终交付物

AI 完成开发后，至少应交付：

1.  完整 `fetch/` Python 项目目录。
2.  `requirements.txt`。
3.  全局配置示例及参数说明。
4.  `note.md` 或 README，说明安装、配置、运行和恢复方式。
5.  统一的运行入口 `control/fetch.py`。
6.  状态文件和错误日志的格式说明。
7.  各数据源输出文件及字段说明。
8.  Master Table 字段字典。
9.  Coverage 和 Disk Usage 报告说明。
10. 自动化测试代码及测试结果。
11. 599 篇 Golden Dataset 回归测试报告。
12. Local 和 Colab Drive 两种运行方式的验证结果。

## 15. 可直接交给 AI 编程代理的总任务说明

请将当前 GP 数据采集 Notebook 工程化为上述 Python 项目。

开发时必须遵循以下优先级：

1.  **保留研究定义和已验证算法。** 不得擅自改变 arXiv v1、首次提交时间
    `t0`、三年引用窗口、C3、IC3、DBLP V3.1.1
    作者匹配和时间处理、OpenReview paper-level mean rating 的定义。
2.  **先迁移，再优化。**
    第一阶段只做模块化和接口整理，不要同时重写核心算法。
3.  **保留和复用已有缓存。** 不得无理由删除或覆盖已验证的 DBLP、Semantic
    Scholar 等缓存。
4.  **建立统一运行机制。**
    使用全局配置、持久化状态、统一日志、错误分类和中控程序。
5.  **保证断点续跑。**
    429、断网、请求失败或运行环境中断后，已成功的数据不得无故重复下载。
6.  **保证数据粒度正确。** 最终 Master Table 必须一篇论文一行，主键为
    `arxiv_id`。
7.  **区分缺失、零值、未匹配和请求失败。** 不允许把 API
    失败伪装成无匹配或零值。
8.  **使用现有 599 篇数据回归测试。**
    在相同缓存和配置下，重构前后关键结果必须一致。
9.  **支持 Local 和 Colab Drive。** 存储路径必须由配置控制，不能将 Colab
    路径硬编码到业务逻辑中。
10. **交付可运行项目。**
    不只提供代码片段；必须提供完整文件、依赖、文档、测试和运行说明。

如果发现现有 Notebook
的某个字段、状态或算法定义不明确，请先标注并报告，不要自行猜测或替换研究定义。

**最终目标不是获得一份看起来更短的代码，而是建立一套可靠、可迁移、可恢复、可复现，并且不会悄悄改变研究数据含义的
GP 数据采集系统。**

另外最后，给我附上requirement.txt和README.md，分别用来一键下载依赖以及对整个项目的解释。

