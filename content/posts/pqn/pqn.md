---
title: "Simplifying Deep Temporal Difference Learning"
date: '2025-02-04T16:09:02+01:00'
summary: "A modern implementation of Deep Q-Network without target networks and replay buffers."
description: "A modern implementation of Deep Q-Network without target networks and replay buffers."
toc: false
readTime: true
autonumber: true
math: true
tags: ["RL", "parallelisation", "jax"]
showTags: false
hideBackToTop: false
fediverse: "@username@instance.url"
hideHeader: true
---


# Abstract 

Q-learning played a foundational role in the field reinforcement learning (RL). However, TD algorithms with off-policy data, such as Q-learning, or nonlinear function approximation like deep neural networks require several additional tricks to stabilise training, primarily a replay buffer and target networks. Unfortunately, the delayed updating of frozen network parameters in the target network harms the sample efficiency and, similarly, the replay buffer introduces memory and implementation overheads. In this paper, we investigate whether it is possible to accelerate and simplify TD training while maintaining its stability. Our key theoretical result demonstrates for the first time that regularisation techniques such as LayerNorm can yield provably convergent TD algorithms without the need for a target network, even with off-policy data. Empirically, we find that online, parallelised sampling enabled by vectorised environments stabilises training without the need of a replay buffer. Motivated by these findings, we propose PQN, our simplified deep online Q-Learning algorithm. Surprisingly, this simple algorithm is competitive with more complex methods like: Rainbow in Atari, R2D2 in Hanabi, QMix in Smax, PPO-RNN in Craftax, and can be up to 50x faster than traditional DQN without sacrificing sample efficiency. In an era where PPO has become the go-to RL algorithm, PQN reestablishes Q-learning as a viable alternative.

---

# Experiments

---

# Conclusions

---

# Try it out
