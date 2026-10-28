# N2空间embedding实验实际状态

本轮已完成CPU筛选，独立复算通过，未达到继续条件。GPU=0，停止当前模型，禁止重复拟合或自动扩大。

8个3585参数模型各150次更新，共1200次；拟合墙钟27.709725秒。480条输入按来源6/2/2，所选种子S1/S2=17、S3/S4=29。Tune中空间S3比旧q提高0.879037个百分点，比最佳对照提高0.707884个百分点，一个来源略降。Hold普通重合度旧q36.878741%、空间37.672828%、简单混合38.198092%。

实际设置、输出、限制见00_READ_ME_FIRST.md；来源数组、权重、逐行q、训练记录及首次失败阶段见raw/。首次0更新启动失败与验证器字段适配均有说明，未重训、未延长原截止。最终独立验证见validation_retry.json，errors为空。

原Target/12k MLP/Ranker/生产源码与历史保留。远端实际目录为/root/autodl-tmp/gaussiangpt_spatial_embedding_20261020，成功拟合子目录run01/experiment02。电脑Chrome原对话URL见WEB_PLAN_NOTES.md。结果包46文件55949640字节已约16:05经原生Chrome上传并发送，网页已开始审阅；完整下一轮回复仍待读取，见WEB_SUBMISSION.md。
