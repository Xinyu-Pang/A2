# Assimilation and Accommodation: Task-Adaptive Hierarchical Abstraction for Solving Web Tasks

## Abstract
Web tasks, which involve processing data from online resources, challenge agents to generalize beyond fixed knowledge to unseen task contexts. Learning from experience, the ability to derive reusable patterns from past tasks, is crucial for improving generalization.
However, existing methods focus on summarizing workflows, \textit{i.e.}, common sub-routines, which may introduce excessive low-level details that distract models. Additionally, the absence of task-specific objectives can lead to inconsistencies between workflows and future task queries, hindering reasoning performance. This paper seeks to mitigate these issues by proposing $A^2$, a framework that derives task-adaptive hierarchical abstraction to enhance web task reasoning. Our approach first extracts general-purpose semantic abstraction from past task-solution pairs. Combined with the next task query, this abstraction forms a task-adaptive episodic abstraction that guides subsequent reasoning. Experiments show that $A^2$ achieves superior performance with competitive cost-efficiency, improving success rates by 0.7\% on Mind2web and 4.6\% on Webarena.


## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Xinyu-Pang/A2.git
cd A2
```

### A2 for Mind2web Dataset
To run $A^2$ on the mind2web dataset:
```bash
cd mind2web
python pipeline.py --load_results 0 --load_scores 1 --if_workflow 1 --top_k_elements 5 --benchmark test_task
```

For comprehensive setup instructions including environment configuration and data preparation, please refer to the detailed documentation in the `./mind2web` directory.


## 📖 Citation

If you find $A^2$ useful, please consider citing our paper:

```bibtex
@inproceedings{pang2025assimilation,
  title={Assimilation and Accommodation: Task-Adaptive Hierarchical Abstraction for Solving Web Tasks},
  author={Pang, Xinyu and Hong, Ruixin and Zhang, Hongming and Zhang, Changshui},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2025},
  pages={14000--14014},
  year={2025}
}
```