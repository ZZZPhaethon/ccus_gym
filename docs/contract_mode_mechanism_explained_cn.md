# 合同模式机制解释

本文档解释纯管道合同模式背后的机制设计。开发时请以 `contract_mode_agent_design_cn.md` 为准；本文用于说明为什么需要这些变量、费用和责任归因。

## 1. 核心思想

当前环境更像月度运行调度：emitter 决定捕集和发送，物理层按当月容量结算。合同模式在此基础上增加一层长期容量承诺：

```text
合同层：
  谁申请多少运输/封存容量？
  pipeline/storage operator 愿意承诺多少能力？
  清算后每个 emitter 获得多少合同容量？

月度层：
  emitter 实际捕集和发送多少？
  pipeline/storage 能否履约？
  合同内、超额、未履约和排放如何结算？
```

这样可以表达英国 CCS 项目中类似 capacity charge + volumetric/flow charge 的结构，同时保留多 emitter 共享管道和共享 storage 时的竞争关系。

## 2. 为什么不使用 Network Charge

第一版不单独建模 `network_charge`。原因是它主要服务于 allowed revenue 回收和监管账务平衡，会让模型复杂化，但不直接改变最小合同博弈。

当前只保留：

```text
capacity_charge
flow_charge
excess_flow_charge
carbon_tax
storage_credit
liability
overflow_penalty
```

如果后续研究重点变成监管收入回收、价格上限或 revenue shortfall，再单独加入 network charge。

## 3. Capacity Charge

`capacity_charge` 是为合同容量付费，不取决于当月实际用了多少。

含义：

```text
我买了每月最多使用 X tCO2 的运输/封存权利；
无论本月实际用了多少，都要为这份权利付固定容量费。
```

这会让 emitter 面临一个权衡：

```text
多订：
  固定费高，但履约保障强，少依赖超额容量。

少订：
  固定费低，但高产或高碳税月份可能不够用。
```

Pipeline/storage operator 也有相应权衡：

```text
多承诺容量：
  capacity revenue 高，但中断/压力高时未履约赔偿风险高。

少承诺容量：
  可靠性更高，但收入和系统封存能力下降。
```

## 4. Flow Charge

`flow_charge` 是合同容量内实际使用时按吨收费。

含义：

```text
capacity_charge 买使用权；
flow_charge 为实际运输/注入的 CO2 付变量费。
```

因此 emitter 的总成本同时包含：

```text
固定容量费 + 实际流量费
```

这可以表达“订了容量但不用会浪费；不用容量则不产生流量费”。

## 5. Excess Flow Charge

`excess_flow_charge` 是超出合同容量的临时使用费。

规则：

```text
先服务合同内容量；
只有存在剩余物理容量时，才接收超额流量；
超额部分被拒收时，operator 不赔偿；
超额流量费固定且高于合同等价成本。
```

为什么要高于合同等价成本？

如果超额费太低，emitter 可能选择：

```text
不订容量，平时只走超额。
```

这会破坏容量合同机制。因此设置：

```text
excess_flow_charge >
    capacity_charge / expected_utilization
  + normal_flow_charge
```

这样只有在确实需要临时超额时，emitter 才会使用 excess。

## 6. Carbon Tax

未捕集 CO2 直接排放并缴纳碳税：

```text
uncaptured = production - captured
carbon_tax_payment = uncaptured * carbon_tax_rate
```

Buffer 只能存放已经捕集的 CO2，不能存未捕集 CO2。

```text
captured CO2 -> emitter buffer / pipeline nomination
uncaptured CO2 -> direct vent / carbon tax
```

因此 `capture_fraction` 和 `send_fraction` 不是同一个决策：

```text
capture_fraction:
  从生产排放中捕集多少。

send_fraction:
  从已捕集库存和本月捕集量中发送多少。
```

捕集但不发送的 CO2 可以暂存在 emitter buffer；如果 buffer 满，则 overflow。

## 7. Storage Credit

`storage_credit` 是对成功注入/封存 CO2 的奖励。

它可以解释为政策补贴或封存信用：

```text
storage_credit_payment = injected_volume * storage_credit_rate
```

第一版可把 credit 给 emitter，因为 emitter 承担捕集和碳税压力。后续也可以分配给 storage operator 或双方共享。

## 8. Liability

合同模式下要区分“谁造成了失败”。

### Emitter 负责

Emitter 对这些结果负责：

```text
未捕集 CO2
超出合同容量的 nomination
超额部分被拒收后导致的 buffer overflow
```

### Pipeline operator 负责

如果 emitter 在合同容量内提交 nomination，而 pipeline 因自身物理可用容量不足无法运输，则 pipeline operator 承担未履约责任：

```text
pipeline_liability =
    pipeline_unserved_contract * pipeline_liability_rate
```

### Storage operator 负责

如果 CO2 已经按合同送达 storage，但 storage 因压力、注入能力或中断无法注入，则 storage operator 承担未履约责任：

```text
storage_liability =
    storage_unserved_contract * storage_liability_rate
```

超额部分不触发 operator liability，因为它没有合同保障。

## 9. Overflow Penalty

`overflow_penalty` 用于 emitter buffer 溢出。

发生条件：

```text
captured CO2 + buffer inventory - accepted sent volume > buffer capacity
```

含义：

```text
捕集了但没有成功送出，也没有足够 buffer 暂存，最终形成 overflow/venting。
```

如果 overflow 来自合同内 operator 未履约，emitter 可以收到 liability compensation；如果来自超额被拒或自身订少容量，则主要由 emitter 承担。

## 10. 合同清算规则

当总请求超过供给时，需要清算合同容量。开发规格中列了几种规则：

| 规则 | 含义 | 激励问题 |
|---|---|---|
| `request_pro_rata` | 按请求量比例分配 | 可能鼓励虚高申请 |
| `fixed_weight` | 按连接权重/设计规模给份额 | 对新增或增长 emitter 不够灵活 |
| `fixed_weight_with_residual` | 固定份额后再分配剩余容量 | 更复杂，但减少 unused capacity |
| `auction` | 按 willingness-to-pay 清算 | 更像市场机制，不完全是受监管容量费 |
| `quota` | 监管/行政配额 | 适合作制度基线 |

本文档不指定默认清算规则。实现时可通过 `contract_clearing_rule` 切换。

## 11. 为什么这个模式有意义

该模式有意义的前提是存在竞争和动态风险：

```text
多个 emitters 共享 pipeline/storage；
pipeline/storage 合同容量有限；
storage pressure 动态影响未来注入能力；
pipeline/storage 可能 disruption；
emitter 有 buffer；
超额流量昂贵且无保障；
operator 对合同内未履约承担 liability。
```

如果系统只有一个 emitter、一个 pipeline、一个 storage，且没有压力动态、中断和容量不足，那么问题会退化成简单成本比较：

```text
捕集封存净成本 vs carbon tax
```

这种情况下不需要 MARL，规则或优化即可。

## 12. 当前不包含的机制

第一版刻意不包含：

```text
network_charge
firm_share / interruptible capacity
off-spec CO2 / 提纯质量惩罚
二级容量交易
动态月度报价
```

这些都可以作为后续扩展模块。

