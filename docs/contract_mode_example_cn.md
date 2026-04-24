# 合同模式运行示例

本文档给出一个最小合同模式的运行例子，用来说明在容量费、流量费、超额流量费和 operator 履约责任存在时，agent 的决策为什么仍然有意义。

示例只考虑：

- 纯管道运输；
- 2 个 emitters；
- 1 条共享 pipeline；
- 1 个共享 storage；
- 不考虑 CO2 off-spec/提纯/质量惩罚；
- 不考虑 `firm_share`；
- 不考虑 `network_charge`。

## 1. 系统设定

```text
Emitters:
  E0
  E1

Infrastructure:
  1 shared pipeline
  1 shared storage

Time:
  1 contract period = 1 year
  1 environment step = 1 month
```

监管/经济参数：

```text
pipeline capacity charge = 4 $/t-capacity/month
storage capacity charge  = 3 $/t-capacity/month

pipeline normal flow charge = 2 $/t
storage normal flow charge  = 3 $/t

pipeline excess flow charge = 12 $/t
storage excess flow charge  = 12 $/t

carbon tax = 30 $/t
storage credit = 5 $/t injected

expected utilization = 0.8
```

合同等价成本：

```text
total capacity charge = 4 + 3 = 7 $/t-capacity/month
total normal flow charge = 2 + 3 = 5 $/t

contract_equivalent_cost_per_t =
    7 / 0.8 + 5
  = 13.75 $/t
```

超额总流量费：

```text
total excess flow charge = 12 + 12 = 24 $/t
```

因此：

```text
total excess flow charge > contract_equivalent_cost_per_t
```

这可以避免 emitter 完全不预订容量、只依赖超额流量。

## 2. 合同层

Pipeline operator 提供：

```text
offered_pipeline_capacity = 100 kt/month
```

Storage operator 提供：

```text
offered_storage_capacity = 90 kt/month
```

Emitters 请求：

```text
E0 requested pipeline capacity = 70 kt/month
E0 requested storage capacity  = 70 kt/month

E1 requested pipeline capacity = 60 kt/month
E1 requested storage capacity  = 60 kt/month
```

总请求：

```text
pipeline demand = 70 + 60 = 130 kt/month
storage demand  = 70 + 60 = 130 kt/month
```

供给低于请求：

```text
pipeline supply = 100 kt/month
storage supply  = 90 kt/month
```

为了示例计算简单，这里采用“按请求量比例分配”清算：

```text
allocated_i = requested_i * supply / total_requested
```

Pipeline 分配：

```text
E0 allocated pipeline =
    70 / 130 * 100
  = 53.8 kt/month

E1 allocated pipeline =
    60 / 130 * 100
  = 46.2 kt/month
```

Storage 分配：

```text
E0 allocated storage =
    70 / 130 * 90
  = 48.5 kt/month

E1 allocated storage =
    60 / 130 * 90
  = 41.5 kt/month
```

每月固定容量费：

```text
E0 capacity payment =
    53.8 * 4 + 48.5 * 3
  = 360.7 k$

E1 capacity payment =
    46.2 * 4 + 41.5 * 3
  = 309.3 k$
```

这一步体现合同层竞争：

```text
E0 和 E1 都想获得共享 pipeline/storage 容量；
总请求超过供给，所以二者都无法获得完整请求量。
```

## 3. 正常月份

假设某月：

```text
E0 production = 70 kt
E1 production = 60 kt

pipeline physical available = 100 kt
storage physical injectable = 90 kt
```

Emitter 月度 action：

```text
E0 capture_fraction = 1.0
E0 send_fraction = 1.0

E1 capture_fraction = 1.0
E1 send_fraction = 1.0
```

E0 捕集并提名发送：

```text
E0 nomination = 70 kt
```

但 E0 storage 合同容量只有：

```text
E0 allocated storage = 48.5 kt
```

因此拆分为：

```text
E0 contracted volume = 48.5 kt
E0 excess volume = 21.5 kt
```

E1：

```text
E1 nomination = 60 kt

E1 contracted volume = 41.5 kt
E1 excess volume = 18.5 kt
```

合同内总量：

```text
48.5 + 41.5 = 90 kt
```

Storage 本月可注入：

```text
storage physical injectable = 90 kt
```

因此合同内 volume 全部服务。

Pipeline 本月可运：

```text
pipeline physical available = 100 kt
```

合同内已使用：

```text
90 kt
```

剩余可用于超额：

```text
100 - 90 = 10 kt
```

超额请求总量：

```text
21.5 + 18.5 = 40 kt
```

假设超额部分也按超额请求比例分配：

```text
E0 accepted excess =
    21.5 / 40 * 10
  = 5.4 kt

E1 accepted excess =
    18.5 / 40 * 10
  = 4.6 kt
```

最终注入：

```text
E0 injected =
    48.5 + 5.4
  = 53.9 kt

E1 injected =
    41.5 + 4.6
  = 46.1 kt
```

未送出或未服务的 captured CO2：

```text
E0 remaining =
    70 - 53.9
  = 16.1 kt

E1 remaining =
    60 - 46.1
  = 13.9 kt
```

这些 remaining CO2：

```text
如果 emitter buffer 有空间，则进入 buffer；
如果 buffer 满，则形成 buffer overflow / venting，并由 emitter 承担后果。
```

正常月份中的决策意义：

```text
如果 buffer 快满，emitter 可能愿意承担昂贵 excess charge；
如果 buffer 充足，emitter 可以少发送，把 CO2 留到未来月份；
如果碳税很高，emitter 更倾向于捕集并尝试使用超额容量。
```

## 4. Storage 履约失败月份

假设另一月 storage 压力升高：

```text
pipeline physical available = 100 kt
storage physical injectable = 60 kt
```

但 storage 合同承诺合计：

```text
E0 allocated storage + E1 allocated storage =
    48.5 + 41.5
  = 90 kt/month
```

Emitters 提交合同内容量内的 nominations：

```text
E0 contracted nomination = 48.5 kt
E1 contracted nomination = 41.5 kt

total contracted nomination = 90 kt
```

Pipeline 可以运输 90 kt：

```text
pipeline physical available = 100 kt
```

但 storage 只能注入：

```text
storage physical injectable = 60 kt
```

因此：

```text
storage unserved contracted volume =
    90 - 60
  = 30 kt
```

这 30 kt 是 storage operator 的合同内未履约责任。

如果：

```text
storage liability rate = 20 $/t
```

则：

```text
storage liability =
    30 kt * 20 $/t
  = 600 k$
```

这部分不应作为 emitter 的普通超额责任，因为 emitters 是在合同容量内提交 nomination。

Storage 的决策意义：

```text
合同层：
  offered_injection_capacity 设得越高，capacity revenue 越高；
  但如果未来压力导致无法注入，就要支付 liability。

月度层：
  injection_fraction 越高，本月未履约赔偿越少；
  但压力可能进一步升高，影响未来可注入能力。
```

## 5. Emitter 少订容量的后果

假设 E1 为了省固定费，在合同层只请求：

```text
E1 requested storage capacity = 30 kt/month
```

但 E1 正常月生产：

```text
E1 production = 60 kt/month
```

即使 storage 系统有余量，E1 合同内也只有：

```text
30 kt/month
```

剩余：

```text
60 - 30 = 30 kt/month
```

属于超额 volume。

处理规则：

```text
如果系统有 spare capacity:
  超额部分可以按 excess_flow_charge 接收；

如果系统没有 spare capacity:
  超额部分被拒收；
  operator 不承担赔偿；
  E1 自己存入 buffer 或承担 overflow/venting 后果。
```

因此 E1 不能总是少订容量。它需要权衡：

```text
少订：
  固定 capacity fee 低；
  但高产、高碳税或拥堵月份风险高。

多订：
  固定 capacity fee 高；
  但合同保障更高，超额和 overflow 风险更低。
```

## 6. 各 Agent 的学习/决策意义

### Emitter

Emitter 的合同层决策：

```text
requested_pipeline_capacity
requested_storage_capacity
```

权衡：

```text
请求多：
  容量保障更强；
  capacity payment 更高；
  用不满时固定费浪费。

请求少：
  固定费更低；
  但需要依赖昂贵且无保障的 excess flow；
  buffer overflow 和 carbon tax 风险更高。
```

Emitter 的月度决策：

```text
capture_fraction
send_fraction
storage_preferences
```

权衡：

```text
多捕集/多发送：
  减少 direct vent 和碳税；
  但可能超合同，支付 excess charge 或被拒收。

少捕集/少发送：
  减少当月运输/封存费用；
  但可能直接排放或增加 buffer 压力。
```

### Pipeline Operator

Pipeline operator 的合同层决策：

```text
offered_pipeline_capacity
```

权衡：

```text
多提供：
  capacity revenue 更高；
  disruption 或维护导致不可用时，liability 风险更高。

少提供：
  履约更稳；
  但收入更低，系统封存能力下降。
```

### Storage Operator

Storage operator 的合同层决策：

```text
offered_injection_capacity
```

权衡：

```text
多承诺：
  capacity revenue 更高；
  压力高或 injectivity 下降时，未履约赔偿风险更高。

少承诺：
  安全余量更大；
  但收入更低，系统封存能力下降。
```

Storage operator 的月度决策：

```text
injection_fraction
```

权衡：

```text
高 injection_fraction：
  本月注入更多，减少未履约赔偿；
  但压力升高，影响未来可注入能力。

低 injection_fraction：
  当前更保守；
  但本月可能产生 non-injection liability。
```

## 7. 什么时候这个模型会变得太简单

如果系统退化为：

```text
1 emitter
1 pipeline
1 storage
无 pressure dynamics
无 disruption
无 buffer limit
无容量不足
excess charge 极高且永远不用
```

那么问题会变成简单成本比较：

```text
捕集封存净成本 vs carbon tax
```

这种情况下不需要学习，规则或优化即可。

要让合同模式有研究意义，至少需要保留：

```text
多个 emitters
共享 pipeline/storage capacity
storage pressure dynamics
operator 未履约 liability
emitter buffer
固定 capacity charge
高价但可用的 excess flow
```

