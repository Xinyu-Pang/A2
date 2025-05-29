# Assimilation and Accommodation: Task-Adaptive Hierarchical Abstraction for Solving Web Tasks

## Abstract
Web tasks, which involve processing data from online resources, challenge agents to generalize beyond fixed knowledge to unseen task contexts. Learning from experience, the ability to derive reusable patterns from past tasks, is crucial for improving generalization.
However, existing methods focus on summarizing workflows, \textit{i.e.}, common sub-routines, which may introduce excessive low-level details that distract models. Additionally, the absence of task-specific objectives can lead to inconsistencies between workflows and future task queries, hindering reasoning performance. This paper seeks to mitigate these issues by proposing $A^2$, a framework that derives task-adaptive hierarchical abstraction to enhance web task reasoning. Our approach first extracts general-purpose semantic abstraction from past task-solution pairs. Combined with the next task query, this abstraction forms a task-adaptive episodic abstraction that guides subsequent reasoning. Experiments show that $A^2$ achieves superior performance with competitive cost-efficiency, improving success rates by 0.7\% on Mind2web and 4.6\% on Webarena.


