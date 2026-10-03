# HARSS 的模型选择：原始研究证据

查阅日期：2026-10-03。讨论对象为手机惯性传感器（inertial sensors）的人体活动识别（human activity recognition, HAR）。本文是文献分析与待验证建议，不表示 HARSS 已选择数据、实现模型或完成实验。

## 判断

建议先建立一维卷积神经网络（one-dimensional convolutional neural network, 1D CNN）或时序卷积网络（temporal convolutional network, TCN）基线（baseline），再比较卷积与注意力（attention）的混合模型（hybrid model）。若任务主要是短窗口内的常见活动，直接采用标准变换器（Transformer）缺乏充分理由；若重点是跨设备、跨佩戴位置和较多样的数据，轻量混合模型值得优先研究。这是结合以下研究的工程判断，不是项目实测结论。

## 直接证据

Ordóñez 与 Roggen（2016）提出 DeepConvLSTM，以卷积提取特征，再用长短期记忆网络（long short-term memory network, LSTM）建模时间动态。在其 OPPORTUNITY 挑战数据上，作者报告比非循环深度网络平均提高约 4%。研究对象含身体佩戴的多模态传感器（multimodal sensors），不能据此认定手机数据或所有任务都需要循环结构。来源：[Sensors 原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC4732148/)，DOI：10.3390/s16010115。

Ek、Portet 与 Lalanda 的 HART 研究显示标准 Transformer 并非总胜出。以下摘录其修订稿表 3 的 UCI 结果；保留原文的 F 分数（F-score）名称，正文没有明确其宏平均（macro-average）或加权平均（weighted-average）方式。

| 模型 | UCI F-score |
| --- | ---: |
| CNN | 94.53 |
| DeepConvLSTM | 95.94 |
| 标准 Transformer | 93.66 |
| HART | 94.49 |
| MobileHART | 97.20 |

MobileHART 增加了卷积层。因此该表更支持卷积与注意力互补，而不是纯注意力普遍优于卷积。本文引用 [arXiv v2（2025-08-23）](https://arxiv.org/abs/2209.11750v2) 的[原文表 3](https://arxiv.org/html/2209.11750#S4.T3)。其题名为 *Transformer-based Models to Deal with Heterogeneous Environments in Human Activity Recognition*；2022 年 v1 的题名不同，不能混用版本数字。

## 其他方式与证据边界

Bai、Kolter 与 Koltun（2018）使用因果卷积（causal convolution）、扩张卷积（dilated convolution）和残差连接（residual connection）构建 TCN，在多种序列任务上优于被比较的循环网络（recurrent neural network, RNN）。它说明较长时间依赖（temporal dependency）不必只能靠 LSTM；这不是手机 HAR 的直接优胜证据。来源：[原文](https://arxiv.org/abs/1803.01271)。

Fawaz 等（2020）的 InceptionTime 采用多尺度卷积（multi-scale convolution）和集成（ensemble），在通用时间序列分类（time series classification）中兼顾精度与扩展能力。它为更强卷积基线提供依据，但通用基准（benchmark）结果不能替代手机 HAR 实验，集成成本也应单独测量。来源：[作者论文](https://arxiv.org/abs/1909.04939)，DOI：10.1007/s10618-020-00710-y。

## 门控循环单元（gated recurrent unit, GRU）的直接证据

Mohsen（2023）以两层纯 GRU 处理 WISDM 的三轴加速度（three-axis acceleration）；数据来自 36 人裤袋中的手机，采样频率（sampling frequency）为 20 Hz，作者报告测试准确率（accuracy）97.08%。正文采用随机 80/20 划分，未明确受试者独立（subject-independent），因此不能据此声称对新用户有相同表现。[原文第 4、6 节](https://link.springer.com/article/10.1007/s11042-023-15571-y)，DOI：10.1007/s11042-023-15571-y。

Tan 等（2022）在 UCI HAR 融合纯 GRU、卷积加 GRU 和接收 561 项人工特征（handcrafted features）的深度神经网络（deep neural network, DNN）。报告的 96.7% 是集成模型（ensemble model）准确率，不能归于纯 GRU。原文序列输入（sequence input）是九通道（nine channels），与上述 HART 的六通道（six channels）不同。[原文第 2 节及表 10](https://ir.lib.cyut.edu.tw/bitstream/310901800/39734/2/electronics-11-00322.pdf)，DOI：10.3390/electronics11030322。

Raja Sekaran 等（2024）的 TCN-GRU 在 UCI HAR、UniMiB SHAR 分别报告 97.25%、93.51% 准确率，并比较纯 GRU；说明 GRU 和卷积混合是可研究方案。数据来自手机不代表模型已在手机上部署（deployment），这些成绩也不能与其他划分和指标的论文直接排名。[PLOS ONE 原文](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0304655)，DOI：10.1371/journal.pone.0304655。

## 评估比模型名称更关键

Dehghani、Sarbishei、Glatard 与 Shihab（2019）发现，重叠窗口（overlapping windows）的收益会受到受试者相关划分的影响；在受试者独立交叉验证（subject-independent cross-validation）下，没有观察到一致的性能增益。不能把“随机窗口划分得分高”直接解释为“对新用户识别好”。来源：[Sensors 原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6891351/)，DOI：10.3390/s19225026。

建议 HARSS 先按受试者划分，再生成窗口；标准化（normalization）统计量只从训练部分估计。固定相同划分、窗口、调参预算（hyperparameter tuning budget），比较小型 CNN、TCN、DeepConvLSTM、轻量混合模型。报告宏平均 F1（macro-averaged F1 score）、每类召回率（per-class recall）、多次随机种子（random seeds）结果、实际设备推理延迟（inference latency）与内存占用（memory footprint）。这些是拟议验证条件，尚未执行。
