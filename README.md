# LLM-Tuning
感谢[LLM-Tuning](https://github.com/beyondguo/LLM-Tuning.git)开源的代码可支持清华大学ChatGLM-6B和ChatGLM2-6B以及百川智能Baichuan-7B的LoRA微调，本项目在此基础上新增模板化生成prompt，以及从历史保存的checkpoint_dir中恢复模型以便继续训练以应对训练中断的场景。

上述两项分别体现在tokenize_dataset_rows.py和baichuan_resume_sft.py脚本里。

部分代码参考[LLaMA-Efficient-Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning)，对这些优秀开源项目表示感谢！
