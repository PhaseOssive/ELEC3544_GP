# GP Target Missing Row Filter

该程序用于执行 Midterm Report 中确定的目标变量缺失处理规则：

**只有当 Y1、Y2、Y3 三个目标同时缺失时，才删除该论文记录。**

如果至少有一个目标可用，则保留该行。Predictor 或 optional metadata 的 Missing 不会触发删除。

## 文件放置

把以下文件放在同一个文件夹：

```text
GP_Target_Missing_Row_Filter/
├── filter_all_targets_missing.py
├── requirements.txt
└── papers_features(4).csv
```

默认输入文件为 `papers_features(4).csv`，默认输出文件为 `papers_features.csv`。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行

```bash
python filter_all_targets_missing.py
```

## 输入文件名不同时

例如输入已经叫 `papers_features.csv`：

```bash
python filter_all_targets_missing.py --input papers_features.csv
```

也可以指定输出路径：

```bash
python filter_all_targets_missing.py \
  --input "papers_features(4).csv" \
  --output "papers_features.csv"
```

相对路径均相对于脚本所在文件夹；也可以传入绝对路径。

## 删除规则

| Target | 字段名 | 说明 |
|---|---|---|
| Y1 | `s2_citation_count_3y` | 三年 Citation Count |
| Y2 | `s2_influential_citation_count_3y` | 三年 Influential Citation Count |
| Y3 | `openreview_mean_rating` | OpenReview Mean Rating |

程序只删除满足以下条件的行：

```text
Y1 is missing AND Y2 is missing AND Y3 is missing
```

其它情况全部保留。特别是：
- Y3 缺失但 Y1 或 Y2 有值：保留。
- Y1 缺失但 Y2 或 Y3 有值：保留。
- Y2 缺失但 Y1 或 Y3 有值：保留。
- 其它 X 或 metadata 字段缺失：不影响是否删除。

## 输出

- `papers_features.csv`：过滤后的主表，保留原来的 25 个字段及其顺序。
- `target_missing_filter_report.txt`：记录输入行数、删除行数、输出行数和三个 Target 的 Missing 数量。

输入文件不会被修改。如果同名输出文件已存在，程序会先将旧输出备份为：

```text
papers_features_before_target_filter.csv
```

如果备份名已存在，会自动追加数字后缀。

## 自动检查

程序会检查：
- 输入表是否有 25 个字段；
- 三个 Target 字段是否存在且为数值或 Missing；
- `paper_id` 是否缺失或重复；
- 输出行数是否与删除行数一致。

检查失败时，程序停止，不会写出过滤后的结果。
