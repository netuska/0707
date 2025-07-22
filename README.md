This paper addresses the environmental impact of cloud computing by focusing on how to efficiently distribute workloads across multiple data centers (DCs) located in different geographic regions. The challenge becomes even more complex when optimizing the cooling systems alongside workload management. To tackle this, the authors introduce Green-DCC, a reinforcement learning-based controller that dynamically coordinates workload allocation and cooling strategies. A key innovation is the use of digital twins—virtual replicas of each DC—which allow the system to simulate strategies before applying them in real conditions. The paper proposes a hierarchical multi-agent reinforcement learning (MARL) approach, since static optimization methods are not sufficient to handle interdependent variables like changing workloads, carbon intensity of local grids, and weather conditions. The Green-DCC architecture has two main layers: a top-level agent that shifts workloads between DCs based on energy availability and carbon intensity, and a low-level agent within each DC that decides when to run tasks and how to manage the liquid cooling system. Non-urgent tasks are queued in a Deferred Task Queue (DTQ) and executed when carbon intensity is low, balancing environmental goals with operational constraints. The authors also include the cost of such actions by factoring in data transfer distances, bandwidth limits, and local resource usage. Among tested algorithms, Proximal Policy Optimization (PPO) performed best, reducing CO₂ emissions to 3435 tons versus 3845 tons in the baseline scenario—outperforming other methods like A2C and APPO. This shows that Green-DCC’s hierarchical MARL is both energy-efficient and practical for real-world cloud infrastructures.
 [Hierarchical Multi-Agent Framework for Carbon-Efficient Liquid-Cooled Data Center Clusters]
 





In this paper  the authors prepose an DC-CFR ; A multi agent reinforcement learnign MARL that handles 3 main operations at once; Cooling Optimization; load shiftting; and battery energy storage. 
the agents work togather using shared state and reward values to learn and adjust to dynamic external conditions ex the battery charging is shiffted to time when grid energy is greener; Loads are redistributed across servers debending on carbon intensity. this framework is performing well as itreduces the carbon emmisions by 14,46% energy use by 14.35% and enrgy costs by 13.69 
[Carbon Footprint Reduction for Sustainable Data Centers in Real-Time]


This paper discuss how to manage the electricity demand in distribution network to decrease the carbon emission by using multi agent Renfocement learning. The authors model the entire system as neworked multi agent constrained Markov decision process and solve it with a safe reinforcement learnign method called consensus multi agent constrained policy optimization 
[Networked Multiagent Safe Reinforcement Learning for Low-carbon Demand Management in Distribution Network]
