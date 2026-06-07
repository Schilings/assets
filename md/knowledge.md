---
title: Knowledge 知识库
layout: default
category: other
---


### 算子

#### 0.Cuda

[【CUDA编程】异步拷贝指令 cp.async 的介绍](https://mp.weixin.qq.com/s/nATI7NBRLOVPL0dCAKqrPg?scene=1&click_id=27)

[CUDA Graph 学习笔记](https://mp.weixin.qq.com/s/9SAW3kgI_G1CzvNM9_aLTg)









[再探 CUDA Graph：核心机制、多图复用以及 Dual AR 模型的统一覆盖优化 ](https://zhuanlan.zhihu.com/p/2017950447520980998)



[torch.compile 技术剖析：PyTorch 2.x 编译系统详解](https://mp.weixin.qq.com/s/kqhkyOgk45adnHHFw7O_GQ)



#### 1. Tensor

[Tensor-002 矩阵乘法优化](https://mp.weixin.qq.com/s/tpzBq_rZdZt3whqw0txCDQ)

[Tensor-003  TensorCore架构](https://mp.weixin.qq.com/s/xdC8SlMQb7O0S3E5VXXpRQ)



#### 2. Gemm



1️⃣[Tensor-103.1: Basic GEMM](https://mp.weixin.qq.com/s/Mt-VRiYtWcqY2EEJN5xQ4g)

2️⃣[Tensor-103.2: Hopper GEMM](https://mp.weixin.qq.com/s/9PKZ9zks1Mx5qaNFqIX3XA)

3️⃣[Tensor-103.3 Hopper Persistent Kernel](https://mp.weixin.qq.com/s/QfxINJ_0ecBzWrnpCI1zTg)

4️⃣[Tensor-103.4 Blackwell GEMM](https://mp.weixin.qq.com/s/HgXWsNa5xkKQ_gmNgmRvbQ)



✅[Blackwell架构学习](https://zhuanlan.zhihu.com/p/1994508526521967481)

🌟[有大神来讲讲 NVIDIA Blackwell 架构吗？ - 手抓饼熊的回答 ](https://www.zhihu.com/question/652991138/answer/2009347256566977753)

1️⃣[Blackwell矩阵乘法：1- 基础介绍](https://zhuanlan.zhihu.com/p/2006030247439638741)

2️⃣[Blackwell矩阵乘法：2- 利用硬件特性优化](https://zhuanlan.zhihu.com/p/2007400131314595305)

3️⃣[Blackwell矩阵乘法：3-实现 85% SOTA 性能背后的优化](https://zhuanlan.zhihu.com/p/2007431045436442536)

4️⃣[Blackwell矩阵乘法：4-突破SOTA](https://zhuanlan.zhihu.com/p/2007441729847055995)

​	

Hopper gemm矩阵乘法优化系列（六）—DEEPGEEM浅析 - neptune的文章 - 知乎
https://zhuanlan.zhihu.com/p/1956080777398837984



deepgemm的fp8 grouped gemm的计算策略 - 赵德柱的文章 - 知乎
https://zhuanlan.zhihu.com/p/1935824134589354647





#### 3. FlashAttention 

https://mp.weixin.qq.com/s/ncCKdKzJlrtluHBUUHDOWQ

[一文深度解析 Flash Attention 3](https://mp.weixin.qq.com/s/ncCKdKzJlrtluHBUUHDOWQ)

[FlashAttention3：再次深度挖掘硬件潜力](https://mp.weixin.qq.com/s/-MdYUoMJAW52jYYgZb_D3A)

[vLLM Triton Attention 后端深度解析](https://mp.weixin.qq.com/s/BbCceV47QJycUDG-hCX7kg?scene=1&click_id=19)

> *Typora 内嵌图片，部署后请替换为线上图片链接*







#### 4. FuseMoe

[SGLang 优化Triton FusedMoE 的一个新技巧](https://mp.weixin.qq.com/s/PxOPFEx-1BstKBhmj9bUtA?scene=1&click_id=39)



#### 5.FlashInfer

[vLLM 推理加速实测：FlashInfer vs FlashAttention，谁才是真王者？](https://mp.weixin.qq.com/s/Z4zMkpZLJGRdIA0rwjB7oQ)



#### 6. CuteDSL



[CuteDSL-1:  Introduction](https://mp.weixin.qq.com/s/X22ucwhTKqEkhF6SrsPdBw)

[CuteDSL-2: 基本操作](https://mp.weixin.qq.com/s/llJ1R9cUGWYFNN0XX0N85Q)

✅[Tensor-103.1: Basic GEMM](https://mp.weixin.qq.com/s/Mt-VRiYtWcqY2EEJN5xQ4g)

✅[Tensor-103.2: Hopper GEMM](https://mp.weixin.qq.com/s/9PKZ9zks1Mx5qaNFqIX3XA)

✅[Tensor-103.3 Hopper Persistent Kernel](https://mp.weixin.qq.com/s/QfxINJ_0ecBzWrnpCI1zTg)

✅[Tensor-103.4 Blackwell GEMM](https://mp.weixin.qq.com/s/HgXWsNa5xkKQ_gmNgmRvbQ)



#### 7. Cutlass

✅[CUTLASS 笔记 (1)：Minimal GEMM Kernel](https://zhuanlan.zhihu.com/p/1937517614084650073)

✅[CUTLASS 笔记 (2)：混合精度 GEMM kernel](https://zhuanlan.zhihu.com/p/1940158874255602181)

✅[CUTLASS 笔记 (3)：Tiled MMA](https://zhuanlan.zhihu.com/p/1950555644814946318)

✅[CUTLASS 笔记 (4)：Tiled Copy](https://zhuanlan.zhihu.com/p/1968745447741972494)

✅[CUTLASS 笔记 (5)：Block MMA](https://zhuanlan.zhihu.com/p/1970162570636816559)

✅[CUTLASS 笔记 (6)：Block Copy](https://zhuanlan.zhihu.com/p/2004627053077627913)



✅[CUTLASS CUTE笔记 (4) - Layout的Composition](https://zhuanlan.zhihu.com/p/28356098779)

Div OP：相当于col major 3 6 2 8 按9为step遍历会命中几个数，这几个数组成的shape

[深入浅出 NVIDIA CuTe：Layout Composition 完整学习笔记](https://zhuanlan.zhihu.com/p/2024861808989676657)
[hello-cute(2)：layout的代数运算 ](https://zhuanlan.zhihu.com/p/2026953818957444858)



#### 8.TileLang

[TileLang深入详解-第一章-GPU 体系结构复习](https://mp.weixin.qq.com/s/16dvqHN8W_4I_XctuAMqUw)



### 并行策略



#### 1. FSDP

[aer(asplos26): FSDP+EP=FSEP（训练-MOE-专家重排）](https://mp.weixin.qq.com/s/h97uEmVozbPD7CHJJlibww)

[PyTorch FSDP Design 详解](https://mp.weixin.qq.com/s/PSY6nnLShRUsMCdn_7TGsw?scene=1&click_id=2)



###  TransformerEngine

#### 1. FP8

✅[从 Transformer Engine 学习 FP8 训练怎么实现](https://mp.weixin.qq.com/s/Ort3hXCFY5X104MykthhbQ)

[从 Transformer Engine 学习 FP8 训练算子该怎么写](https://mp.weixin.qq.com/s/QvZ4s666WJCkzoYUKNz38g)



[一文带你玩转 FP8 RL 全流程训练](https://mp.weixin.qq.com/s/Z_6Xx8EaSfg6Qbm3tx82Ag)

#### 2. FP4

[FP4 训练实现机制](https://mp.weixin.qq.com/s/Z9TQJYjGsiIkelKV2uPIdw)



### Moe

[【论文阅读】使用 Megatron Core 进行混合专家模型的可扩展训练](https://zhuanlan.zhihu.com/p/2014769519407682736)

[NVIDIA技术沙龙 《大规模EP优化：PD分离MoE并行方式》](https://zhuanlan.zhihu.com/p/1944769427154371625)



### Megatron

#### 1. Hybrid EP

- A new backend implementation using TMA instructions for minimal SM usage and larger NVLink domain support
  一个新的后端实现，使用 TMA 指令，实现最小的 SM 使用和更大的 NVLink 域支持
- Fine-grained communication-computation overlap for single-batch scenarios
  单批次场景下的细粒度通信-计算重叠
- PCIe kernel support for non-NVLink environments
  非 NVLink 环境的 PCIe 内核支持
- NVFP4 data type support
  NVFP4 数据类型支持

[英伟达Megatron-Core MoE 学习（四）：MoE GPU通信优化](https://mp.weixin.qq.com/s/ChDh7c22reorsJgK_gGOuQ)

[Hybrid-EP - 面向混合专家模型训练的通信优化方案](https://mp.weixin.qq.com/s/D1gxbGeSf2rWvuXRIoTuow)

[[翻译 & 解读] 在MoE训练中用 Hybrid-EP 优化通信](https://zhuanlan.zhihu.com/p/2009695597112865631?share_code=1r0A6rATHAG7V&utm_psn=2027825224226187086)



#### 2. DeepEP

[DeepEP](https://github.com/deepseek-ai/DeepEP)

✅[kernel 笔记: DeepEP (1) 预备知识 - 大卫的文章 - 知乎](https://zhuanlan.zhihu.com/p/1928161639397586106)

NVSHMEM专题：[ 谈谈deepEP中的NVSHMEM - 知乎](https://zhuanlan.zhihu.com/p/1898141047164507218)

[GPU 原生通信黑科技！带你三步上手 DeepEP 加速专家模型](https://mp.weixin.qq.com/s/3ThnX9cJUgyOE4jxMcapAg?scene=1&click_id=4)

> Intra-Node

✨[乱谈kernel之DeepEP Intranode - 是小肖啊的文章 - 知乎](https://zhuanlan.zhihu.com/p/1998516883851352048)

✅[一文彻底看懂DeepEP（1）：intranode - BoLi2001的文章 - 知乎](https://zhuanlan.zhihu.com/p/1935094821804040824)

[一文彻底看懂DeepEP（2）：low latency - BoLi2001的文章 - 知乎](https://zhuanlan.zhihu.com/p/1935108598784058926)

> Inter-Node

✅[为 MoE 而生的分布式通信框架：DeepEP 详解](https://mp.weixin.qq.com/s/XoUDCbm4g-hqE_-5DN7zKg?scene=1&click_id=5) (inter-node)

✅[DeepEP 邮局（二）：L 形路由跨节点通信 - Sonder的文章 - 知乎](https://zhuanlan.zhihu.com/p/1988670519017492484)

✨[DeepEP实现分析1——overview](https://mp.weixin.qq.com/s/QVkNEB2knvmXe4OtuXsV1g)

✅[DeepEP源码分析1——internode notify_dispatch](https://mp.weixin.qq.com/s/CL83S7xAmfUMINBkk23nuw)

✅有图：[理解DeepEP源码和节点通信逻辑 - 知乎](https://zhuanlan.zhihu.com/p/1890067712996270654)(inter-node)

✅有图：[DeepEP Dispatch/Combine 图示 - 知乎](https://zhuanlan.zhihu.com/p/29273768638)

✅[【zartbot】分析一下EP并行和DeepSeek开源的DeepEP代码 - 微信](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s/1Cz7oQbVkPMam3eoKQWz0w)





#### 3. UCCL-EP

[UCCL-EP: Portable Expert-Parallel Communication — Full Results](https://uccl-project.github.io/posts/uccl-ep-full/)

[UCCL-EP: 怎样把DeepEP跑在"奇奇怪怪"的非Nvidia硬件上?](https://zhuanlan.zhihu.com/p/2025935535215055709?share_code=wbgGZHCY830X&utm_psn=2027828476288188619)



#### 4. 混合精度训练 



FP16 BF16 混合精度训练 loss scale

https://mp.weixin.qq.com/s/Elx7kRN09OsZ-dhKPC53mA

✅[图解大模型训练系列之：Megatron源码解读3，分布式混合精度训练](https://mp.weixin.qq.com/s/Elx7kRN09OsZ-dhKPC53mA?scene=1&click_id=34)

#### 5. Fine-grianed Offload

✅[大模型训练的高效内存解决方案：流水线感知的细粒度激活卸载，实现显存开销与吞吐性能的联合最优](https://mp.weixin.qq.com/s/-7YEig9nAte5DfDthn3oiQ?scene=1)

🌟[Fine-Grained Offload HTML](/h5/fine-grained-offload.html) | [Fine-Grained Offload MD](/md/fine-grained-offload-detail.html)





#### 6. Pipeline Parallel

[AI Infra论文阅读之将流水线并行气泡几乎降到零（附基于Meagtron-LM的ZB-H1开源代码实现解读）](https://mp.weixin.qq.com/s/PXjYm9dN8C9B8svMQ7nOvw?scene=1)

[PP->VPP->ZeroBubblePP->deepseekv3 dualPipe，对PP bubble的极致压缩](https://zhuanlan.zhihu.com/p/26559590326?utm_source=wechat_session&utm_medium=social&s_r=0)



#### 7. Sequence Parallel

[SP 分析 HTML](/h5/sp-analysis.html)





### DeepSeek

[DeepSeek-V4 百万级上下文相关代码到实现](https://mp.weixin.qq.com/s/dCKqtINt83x_Vw28bBhHJA)

[DeepSeek V4-vLLM预览](https://mp.weixin.qq.com/s/KFuU_ghQewXYckQGP5wuCA)



#### 模型结构 

✅[详细谈谈DeepSeek MoE相关的技术发展](https://mp.weixin.qq.com/s/sPUMkaFAiQBTL_U66bKEzA)



💗[【手撕 mHC】DeepSeek残差链接mHC进化之路（详解+附代码）](https://mp.weixin.qq.com/s/V99X4a-pOzM_3OVKDNWqhQ?scene=1)

[浅谈 DeepSeek mHC 一种可能的加速方案：小矩阵遇上牛顿法](https://mp.weixin.qq.com/s/eMiJ2LDJ6a14C33Xu60EYw)

✅[DeepSeek-V4 mHC：从原理到源码](https://mp.weixin.qq.com/s/T_52fky_xb-9SNPLSAp22w)

[DeepSeek-V4详细分析(1): 算法和模型结构](https://mp.weixin.qq.com/s/F-0_bbwvQjlYaHVFW_uPNw)



[Muon优化器深度解析及与AdamW对比](https://mp.weixin.qq.com/s/HuXLPDchi_eC01VydKOy6g)



#### Infra

[DeepSeek-V4 通用基础设施软硬件协同设计：细粒度专家并行最高实现 1.96 倍加速](https://mp.weixin.qq.com/s/m0tHhBbVUjQvssO-qVgENw)

[最高砍掉90%计算成本！揭开DeepSeek-V4超长上下文的高效密码与未解难题](https://mp.weixin.qq.com/s/lWGZ3TcPj5_mb8HUDrsxsg)

[DeepSeek 开源内部训练算子库 TileKernels：MoE 路由到量化融合，全部用 TileLang 实现](https://mp.weixin.qq.com/s/Jj5dAKx3NIaJVQfpJjNhGA)

[Infra细节太多，DeepSeek V4技术报告阅读](https://mp.weixin.qq.com/s/U3ifxtMTcX7Rd8UzlUHlKg)





[Agent写的一个高性能Host-to-Device 传输库](https://mp.weixin.qq.com/s/zs5MLHvsBSsyh9EPrzhwWw)

#### MegaMoe

[DeepGEMM #304浅析：Mega MoE、FP4 Indexer 与全面架构升级](https://mp.weixin.qq.com/s/I3ijRMTp1Kz_iK7xqDJOkw)

✅[DeepSeek-V4详细分析(2): MegaMoE](https://mp.weixin.qq.com/s/S-ej9ybT3sbFA8dqHLZafg)

[MoE 训练提速最高 38%！字节 Seed 开源 UniEP：首个训练级 MegaKernel 架构，重新定义专家并行训练的性能天花板！](https://mp.weixin.qq.com/s/6vzKPwytnPDaVJNpi15m5g)





#### DeepEP v2

✅[从零开始的通信计算overlap【第一章】](https://zhuanlan.zhihu.com/p/2011564057396809841)

✅[从零开始的通信计算overlap【第二章】大模型通信基础 2.1：通信硬件拓扑](https://zhuanlan.zhihu.com/p/2028907020917449344)

✅[从零开始的通信计算overlap【第二章】大模型通信基础2.2 RDMA 核心概念](https://zhuanlan.zhihu.com/p/2028907599861495146)

✅[从零开始的通信计算overlap【第二章】大模型通信基础 2.3 机内数据搬运](https://zhuanlan.zhihu.com/p/2028907936030704604)

✅[从零开始的通信计算overlap【第二章】大模型通信基础 2.4 机间数据搬运](https://zhuanlan.zhihu.com/p/2028908577935336722)



✅[NVIDIA NCCL 2.28重磅发布：GPU 直接通信时代，CPU 变成 "配角"！](https://mp.weixin.qq.com/s/a4Tg0a9qCpThq6Dy7WHgDg)

💗[NVIDIA NCCL GIN 论文发布！从论文到代码再探 NCCL 2.28 GPU-Initiated Networking](https://mp.weixin.qq.com/s/peSA0QsnROYyWPDqhYd7Ng)

💗[DeepEPv2分析(1)](https://mp.weixin.qq.com/s/Zm4vIvqUVxiEZfhNkw_Q_g)

💗[DeepEPv2分析(2)-EP Overview](https://mp.weixin.qq.com/s/BgskWRHh98QX2TPZKcPpPQ)

💗[DeepEPv2分析(3)-EP Direct Dispatch/Combine Kernel](https://mp.weixin.qq.com/s/LbZdSJMFCUj6tojmMyH5gg)

💗[DeepEPv2分析(4)-EP Hybrid Dispatch Combine Kernel](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247498255&idx=2&sn=4911355e9cabc2f7303e148d247da2ea&chksm=f82064bfbb165131820950711c84393b8971f28b746e3c803761ac152c0c0e37bba0302c239a&scene=126&sessionid=1778163054&subscene=227&clicktime=1778163100&enterid=1778163100#rd)



#### DeepGEMM

[【DeepSeek V4】DeepGEMM 为什么不支持你的显卡？从 PyTorch 到 CUDA 的无可奈何](https://mp.weixin.qq.com/s/e1GKi53IociZnfHX1Ie9_w)



#### DualPipe

![DualPipe](/images/9a914356d7f48eee79cee286fc514d16.jpg)

[mlc-ai/Pith-Train用1万行代码，把生产级MoE训练栈的4D并行、DualPipeV重叠调度与FP8算子，压缩成可读、可改、可验证的Python工程](https://mp.weixin.qq.com/s/jHSGorh2xSVTvkY6Fll8SA)

[GraphPP：基于计算图的流水线并行，零气泡+极致重叠，训练提速 70%+](https://mp.weixin.qq.com/s/CWaYd59MX9CJ6MCdcmie4g)

### SGLang

[基于 mini-sglang 学习大模型推理关键功能](https://mp.weixin.qq.com/s/pjSQMpHVAMXZh0uxM9ksNw)

[【短文】大模型推理加速：从面向对象到面向数据设计](https://mp.weixin.qq.com/s/-BdKxv3NY1mDWjL_iJDMWA)

✅[SGLang 分布式集群模式概览](https://mp.weixin.qq.com/s/7g472bLmKr9JyrMuJgPWJw)

✅[SGLang 的 TP 模式浅析](https://mp.weixin.qq.com/s/HSVQ3WJtLAAbkRHY9jRu9Q)

✅[SGLang 的 PP 模式浅析](https://mp.weixin.qq.com/s/HrFY_uz4U5GFRNo0U2E40A)

✅[SGLang 的 DP Attention 模式浅析](https://mp.weixin.qq.com/s/1CaGUQcWj9DFUyDS9jylLw)

 

✅[大模型推理加速：Overlap Scheduling 的深入剖析与性能权衡艺术](https://mp.weixin.qq.com/s/lWrDXk0VQA3bYC9ZnF2GjQ)

[降低RL训推共卡开销：SGLang/vLLM的无缝切换实现与分析](https://mp.weixin.qq.com/s/hgKCt7OY-8GSo2Mdwsgb3g?scene=1&click_id=38)


[聊一下 sglang mixed chunk的技术 - 匆匆那年的文章 - 知乎](https://zhuanlan.zhihu.com/p/1980240042275382681)

✅[从 KV Cache 到 Zero Overhead Scheduling，一文读懂 SGLang 的调度巧思](https://mp.weixin.qq.com/s/-O5W_4CGD0XJMAtHckn3nw)



[sgl-kernel MoE Align Block Size Kernel 优化过程解析](https://mp.weixin.qq.com/s/37hIuVCN3SP_vo2OfHWVVw)



### vLLM

💗[LLM推理优化-vLLM TP并行](https://mp.weixin.qq.com/s/V27UNfTAzokpMV3EolNFUA)

💗[LLM推理优化-vLLM SP并行](https://mp.weixin.qq.com/s/dztAI5PciugDzD2yv8a2_g)

💗[LLM推理优化-vLLM Async TP](/h5/sp.html)

💗[vLLM Async SP — Symmetric Memory 深度解析](/h5/async-sp-symm-mem.html)

✅[vllm并行策略之DCP(Decode Context Parallel)](https://zhuanlan.zhihu.com/p/2020086868914499979?share_code=qFykZcphGY0L&utm_psn=2027673704973214084)

✅[LLM推理优化-vLLM CP并行](https://mp.weixin.qq.com/s/qZj8ni7rvydTW7joVYdMFw)

✅[LLM推理优化-vLLM DP并行](https://mp.weixin.qq.com/s/AWIDU0g-NKDu7SpP9BQ1Xg)

[vLLM Part2.3-Engine核心代码解读](https://mp.weixin.qq.com/s/q4Al5uQMHvHrU1dvu-K9Qg)



[基于 nano-vLLM 学习大模型推理关键功能](https://mp.weixin.qq.com/s/6mAZ49iP1SCKt5ZdWf6ErQ)

[Model Runner V2：更模块化、更快速的 vLLM 执行核心](https://mp.weixin.qq.com/s/DrZLBvKEfevQaMpa-eXscQ)

[vLLM 权重加载机制全解析：从挑战到理想架构](https://mp.weixin.qq.com/s/u_OJe3Y6EIJdo41CH9evIA)



[vLLM PIECEWISE CUDA Graph 技术学习笔记](https://mp.weixin.qq.com/s/JCzBtIAMVOiNtisHxZBTUw)

[LLM推理效率瓶颈突破：数据并行负载均衡(DPLB)浅析](https://mp.weixin.qq.com/s/mPtyZoYlQgthoAqeQywkmA?scene=1&click_id=58)

[拆解vLLM DP技术栈：特性原理、演进瓶颈与优化方案](https://mp.weixin.qq.com/s/cJi5NN0FpKzf4DAJIlRl3Q?scene=1&click_id=60)



[vLLM 权重加载机制全解析：从挑战到理想架构](https://mp.weixin.qq.com/s/u_OJe3Y6EIJdo41CH9evIA)





### PD 分离

[SGLang PD分离架构深度解析：Prefill、Decode与Router的协同之道](https://mp.weixin.qq.com/s/RYQ547Gxktza0_PPhfn_Iw)

【探秘Transformer系列之（26）--- KV Cache优化 之 PD分离or合并】https://mp.weixin.qq.com/s/akM8wUsaVNstAVYJDLC4IQ

【Mooncake 架构概览：以 KVCache 为中心的高效 LLM 推理系统设计】https://mp.weixin.qq.com/s/BUGmmvLZ7TrmwleDWsb5SA

https://mp.weixin.qq.com/s/BUGmmvLZ7TrmwleDWsb5SA?scene=1&click_id=22





### 推测解码

[大模型推理加速：EAGLE-3介绍](https://mp.weixin.qq.com/s/aUG6imjEVWA_QrNvsklB4g?scene=1&click_id=42)

[探秘Transformer系列之（33）--- DeepSeek MTP](https://mp.weixin.qq.com/s/OskEW7K602HNPm2PUBnSRg?scene=1&click_id=43)

[探秘Transformer系列之（31）--- Medusa](https://mp.weixin.qq.com/s/DfGLmys8b3vydCnSKU2ENw?scene=1&click_id=44)

[探秘Transformer系列之（30）--- 投机解码](https://mp.weixin.qq.com/s/sfI9_ohC0DCd05Z4gxbBXA?scene=1&click_id=45)



[投机解码之EAGLE：轻量级草稿模型实现3-4倍推理加速](https://mp.weixin.qq.com/s/8GHrSIeOJQVFAbSQ7JGi_Q)





### 强化学习

#### 1. 算法

 [详解策略梯度算法.pdf] — *本地文件，部署后不可用*

 [详解近端策略优化.pdf] — *本地文件，部署后不可用*

从TRPO到PPO（理论分析与数学证明） https://www.cnblogs.com/xingzheai/p/16565686.html

策略梯度 https://www.cnblogs.com/xingzheai/p/15826847.html

ppo https://www.cnblogs.com/xingzheai/p/15931681.html)

![RL](/images/33e2cedb0ac5a1e573917de1d32d485e.png)

[RL101:从SFT到PPO](https://mp.weixin.qq.com/s/YA5akqX4oZfc2MScjzRDDQ)

[RL102: 从ORM到PRM, Reasoning模型诞生](https://mp.weixin.qq.com/s/pjGDnx97sZ62TVjJ9wUQ5g)

[RL103: RLVR and RLAIF](https://mp.weixin.qq.com/s/41WH9ztVLlYPMeSqauiE9Q)

[类PPO强化学习三部曲：GRPO简化→DAPO修正→GSPO全面进化](https://mp.weixin.qq.com/s/w-DBnIcgHfl7F5vSFsAbxQ?from=singlemessage&scene=1&subscene=317&sessionid=1771998340&clicktime=1772012569&enterid=1772012569&ascene=1&fasttmpl_type=0&fasttmpl_fullversion=8131860-zh_CN-zip&fasttmpl_flag=0&realreporttime=1772012569086&click_id=36)

[大模型强化学习算法的演进与对比：从PPO、GRPO、DAPO到GSPO、SAPO](https://mp.weixin.qq.com/s/GiUB8WIbm6tk-6_BSbvvJQ)

#### 2. 框架

[图解Infra视角下的强化学习性能优化](https://mp.weixin.qq.com/s/onW4d3e-SCgO_z_16JgWEA)

### 





#### Relax

[小红书 Relax 开源发布：面向全模态 Agentic 的异步 RL 训练引擎](https://mp.weixin.qq.com/s/u0-tZEWTPX4Jh9y9EbVJ1g)

### 知识

【手撕 mHC】详解DeepSeek残差链接mHC进化之路（超长文、附代码） - 小冬瓜AIGC的文章 - 知乎
https://zhuanlan.zhihu.com/p/1990683672337223894



