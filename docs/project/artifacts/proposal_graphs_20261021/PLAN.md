# 三种提案侧专用图实测

为Ranker+MLP、EAGLE Draft和Medusa LowRank预测头分别捕获前向图，保留现有权重、FP32精度和采样规则。EAGLE普通历史重算版与KV复用版各自配对。共同Target使用既有Graph+BMM实现。

每种方法比较普通提案与图提案；另报告各方法相对同优化Target的完整解码倍率。首次建图、KV分配、前缀处理、提案、验证、恢复与采样均计入外层同步墙钟。权重加载另计。各执行臂持有独立图，预热图丢弃；轮换执行次序。

先验证同输入数值与同种子输出、接受计数，再以10开发来源、64/256前缀、单种子、512新token运行主要比较。报告共同满长度、无新增图请求子集及图实际调用数量。测量无收益也完整记录，不训练或替换生产默认。

源码：patches/proposal_graphs_20261021；远端：/root/autodl-tmp/gaussiangpt_proposal_graphs_20261021。原工程/root/workspace/GaussianGPT与历史实验保留。三名原Luna/max子代理分别负责MLP/Medusa、EAGLE与独立复算，主代理整合并独占GPU。
