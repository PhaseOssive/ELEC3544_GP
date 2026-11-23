# GP DBLP Treated Feature Merge

这个程序用于把 DBLP Recovery **完成 Missing Treatment 后**的正式团队特征，合并回 `papers_features.csv`。

> **默认使用 `treated_team_features.csv`，不要使用 `final_team_features.csv` 作为最终合并输入。**
> `final_team_features.csv` 是 Missing Treatment 前的 Recovery 结果，适合保留作审计；`treated_team_features.csv` 才是按项目 Missing Treatment 规则处理后的结果。

## 文件放置

把以下三个文件放在同一个文件夹：

```text
your-folder/
├── merge_dblp_features.py
├── papers_features.csv
└── treated_team_features.csv
```

程序默认从脚本所在文件夹读取两个 CSV，并在同一文件夹输出 `papers_features.csv`。

## 运行环境

需要 Python 3.9 或更新版本，以及 pandas、numpy。

安装依赖：

```bash
pip install -r requirements.txt
```

运行：

```bash
python merge_dblp_features.py
```

## 默认输入和输出

- 主表：`papers_features.csv`
- DBLP Missing Treatment 后的表：`treated_team_features.csv`
- 输出：`papers_features.csv`

默认输出会覆盖主表。为避免误操作，程序会在覆盖前自动创建备份：

```text
papers_features_before_dblp_merge.csv
```

如果该备份已存在，会自动在文件名后添加 `_2`、`_3` 等数字。

## 如果 CSV 文件名不同

可以通过命令行指定文件名。例如：

```bash
python merge_dblp_features.py \
  --base "papers_features(3).csv" \
  --dblp "treated_team_features(1).csv" \
  --output "papers_features_merged.csv"
```

相对路径均相对于脚本所在文件夹。也可以使用绝对路径。

## 合并规则

1. 以 `papers_features.csv` 为主表，保留其原始 25 个字段和字段顺序。
2. 通过主表的 `paper_id` 与 Treatment 表的 `arxiv_id` 匹配，不依赖两张表的行顺序。
3. 对新式 arXiv ID 的序号部分进行五位规范化。例如 `1701.0064` 会规范化为 `1701.00640`。最终输出的 `paper_id` 使用 Treatment 表中的规范 ID。
4. 从 `treated_team_features.csv` 取回四个正式 DBLP 特征：
   - `dblp_team_mean_publication_count_pre_t0`
   - `dblp_team_max_publication_count_pre_t0`
   - `dblp_team_mean_academic_age_pre_t0`
   - `dblp_team_max_academic_age_pre_t0`
5. 四个 DBLP 特征以 `treated_team_features.csv` 为准，包括保留其中的 Missing Treatment 结果和仍然无法可靠恢复的 Missing；不会回填主表中的旧 DBLP 数值。
6. Treatment 表中的其他审计字段不会添加到主表。主表其他 21 个字段保持不变，最终仍为 25 个字段。

## 自动检查

程序会在写出结果前检查：

- 两个文件的行数；
- ID 是否缺失或重复；
- 规范化后两个文件的 ID 集合是否完全一致；
- 四个 DBLP 字段是否为有效数值；
- DBLP 数值是否为负数；
- Mean 是否大于 Max；
- Mean/Max 的缺失状态是否不一致。

任何关键检查失败时，程序会停止且不会写出最终结果。
