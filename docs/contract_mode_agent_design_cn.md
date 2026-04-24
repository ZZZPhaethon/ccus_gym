# 合同模式 Agent 开发规格

本文档定义纯管道合同模式的最小可实现版本。范围：无 `network_charge`、无 `firm_share`、无 CO2 off-spec/提纯质量机制。合同容量按月度容量表示，单位为 `MtCO2/month`。

## 1. Agent

| Agent | 合同层 action | 月度 action |
|---|---|---|
| `emitter_i` | `[requested_pipeline_capacity, requested_storage_capacity_0, ...]` | `[storage_preferences..., capture_fraction, send_fraction]` |
| `pipeline_operator` | `[offered_pipeline_capacity]`，也可先规则给定 | 无 |
| `storage_j` | `[offered_injection_capacity]` | `[injection_fraction]` |
| `regulator` | 规则模块 | 规则模块 |

单 storage 时删除 `storage_preferences`，emitter 月度 action 为：

```text
[capture_fraction, send_fraction]
```

## 2. Parameters

Regulator/config 提供：

```text
carbon_tax_rate
pipeline_capacity_charge_rate
pipeline_flow_charge_rate
pipeline_excess_flow_charge_rate
storage_capacity_charge_rate[sid]
storage_flow_charge_rate[sid]
storage_excess_flow_charge_rate[sid]
storage_credit_rate
pipeline_liability_rate
storage_liability_rate[sid]
overflow_penalty_rate
expected_utilization
max_excess_ratio
contract_clearing_rule
```

固定约束：

```text
pipeline_excess_flow_charge_rate
  > pipeline_capacity_charge_rate / expected_utilization
    + pipeline_flow_charge_rate

storage_excess_flow_charge_rate[sid]
  > storage_capacity_charge_rate[sid] / expected_utilization
    + storage_flow_charge_rate[sid]
```

## 3. Contract State

合同清算后保存：

```text
allocated_pipeline_capacity[eid]
allocated_storage_capacity[eid, sid]
```

可选保存：

```text
requested_pipeline_capacity[eid]
requested_storage_capacity[eid, sid]
offered_pipeline_capacity
offered_injection_capacity[sid]
```

## 4. Clearing

如果总请求不超过供给：

```text
allocated = requested
```

如果请求超过供给，`contract_clearing_rule` 选择一种：

| Rule | Formula / Logic |
|---|---|
| `request_pro_rata` | `allocated_i = requested_i * supply / sum(requested)` |
| `fixed_weight` | `entitlement_i = supply * weight_i / sum(weight)`；`allocated_i = min(requested_i, entitlement_i)` |
| `fixed_weight_with_residual` | 先按 `fixed_weight` 分配；剩余容量按未满足请求 `unmet_i` 再分配 |
| `auction` | 按 bid / willingness_to_pay 清算 |
| `quota` | 使用外部给定 `regulator_defined_quota_i` |

Storage 容量在每个 `sid` 内独立清算。

## 5. Monthly Physics Interface

Emitter 月度物理量：

```text
captured = production * capture_fraction
uncaptured = production - captured
available_to_send = emitter_buffer + captured
nomination = available_to_send * send_fraction
```

未捕集 CO2 直接排放：

```text
direct_vent = uncaptured
carbon_tax_payment = direct_vent * carbon_tax_rate
```

Buffer 只存捕集后的 CO2：

```text
sent = accepted_contract_flow + accepted_excess_flow
buffer_next = emitter_buffer + captured - sent
buffer_overflow = max(0, buffer_next - buffer_capacity)
emitter_buffer = min(buffer_next, buffer_capacity)
```

## 6. Monthly Settlement

### 6.1 Split Nomination

对每个 emitter：

```text
contract_pipeline_nomination =
    min(nomination, allocated_pipeline_capacity[eid])

excess_pipeline_nomination =
    max(0, nomination - allocated_pipeline_capacity[eid])
```

对 storage 同理，按 `storage_preferences` 或单 storage 分配后计算：

```text
contract_storage_nomination[eid, sid] =
    min(storage_nomination[eid, sid], allocated_storage_capacity[eid, sid])

excess_storage_nomination[eid, sid] =
    max(0, storage_nomination[eid, sid] - allocated_storage_capacity[eid, sid])
```

可限制超额请求：

```text
excess_nomination <= max_excess_ratio * allocated_capacity
```

### 6.2 Pipeline Allocation

服务顺序：

```text
1. contract pipeline nominations
2. excess pipeline nominations
```

合同内物理容量不足时，按选定月度分配规则削减，最小版本可用按合同 nomination 比例：

```text
accepted_contract_pipeline[eid] =
    contract_nomination[eid] * available_pipeline_capacity / total_contract_nomination
```

如果合同内已服务后还有剩余：

```text
remaining_pipeline_capacity =
    available_pipeline_capacity - sum(accepted_contract_pipeline)
```

再服务 excess：

```text
accepted_excess_pipeline[eid] =
    excess_nomination[eid] * remaining_pipeline_capacity / total_excess_nomination
```

合同内未运输量：

```text
pipeline_unserved_contract[eid] =
    contract_pipeline_nomination[eid] - accepted_contract_pipeline[eid]
```

超额未服务不触发 operator liability。

### 6.3 Storage Injection

对每个 storage：

```text
physical_injectable =
    storage.get_max_injectable() * injection_fraction
```

服务顺序：

```text
1. contract delivered volume
2. excess delivered volume
```

合同内注入不足：

```text
storage_unserved_contract[eid, sid] =
    contract_delivered[eid, sid] - injected_contract[eid, sid]
```

超额未注入不触发 operator liability。

## 7. Payments

容量费：

```text
pipeline_capacity_payment[eid] =
    allocated_pipeline_capacity[eid] * pipeline_capacity_charge_rate

storage_capacity_payment[eid, sid] =
    allocated_storage_capacity[eid, sid] * storage_capacity_charge_rate[sid]
```

合同内流量费：

```text
pipeline_flow_payment[eid] =
    accepted_contract_pipeline[eid] * pipeline_flow_charge_rate

storage_flow_payment[eid, sid] =
    injected_contract[eid, sid] * storage_flow_charge_rate[sid]
```

超额流量费：

```text
pipeline_excess_payment[eid] =
    accepted_excess_pipeline[eid] * pipeline_excess_flow_charge_rate

storage_excess_payment[eid, sid] =
    injected_excess[eid, sid] * storage_excess_flow_charge_rate[sid]
```

Operator liability / emitter compensation：

```text
pipeline_liability_payment[eid] =
    pipeline_unserved_contract[eid] * pipeline_liability_rate

storage_liability_payment[eid, sid] =
    storage_unserved_contract[eid, sid] * storage_liability_rate[sid]
```

Storage credit：

```text
storage_credit_payment[eid] =
    total_injected_for_emitter[eid] * storage_credit_rate
```

Overflow penalty：

```text
buffer_overflow_penalty[eid] =
    buffer_overflow[eid] * overflow_penalty_rate
```

## 8. Reward

Emitter:

```text
r_emitter =
    + storage_credit_payment
    + pipeline_liability_payment
    + storage_liability_payment
    - capture_cost
    - electricity_cost
    - pipeline_capacity_payment
    - storage_capacity_payment
    - pipeline_flow_payment
    - storage_flow_payment
    - pipeline_excess_payment
    - storage_excess_payment
    - carbon_tax_payment
    - buffer_overflow_penalty
    + w_global * system_reward
```

Pipeline operator:

```text
r_pipeline =
    + sum(pipeline_capacity_payment)
    + sum(pipeline_flow_payment)
    + sum(pipeline_excess_payment)
    - pipeline_operating_cost
    - repair_or_disruption_cost
    - sum(pipeline_liability_payment)
    - congestion_or_reliability_cost
    + w_global * system_reward
```

Storage operator:

```text
r_storage[sid] =
    + sum_i storage_capacity_payment[i, sid]
    + sum_i storage_flow_payment[i, sid]
    + sum_i storage_excess_payment[i, sid]
    - injection_operating_cost
    - pressure_risk_cost
    - pressure_violation_penalty
    - sum_i storage_liability_payment[i, sid]
    + w_global * system_reward
```

System reward 最小版本：

```text
system_reward =
    stored_weight * total_injected
  - vented_weight * total_vented
  - pressure_weight * pressure_violation_penalty
  - energy_weight * total_energy_use
```

## 9. Observations

Emitter observation:

```text
own requested capacities
own allocated capacities
own buffer fraction
own last accepted contract flow
own last accepted excess flow
own last rejected/excess-unserved flow
own last capacity utilization
pipeline utilization/scarcity/disruption signal
connected storage pressure/max injectable/disruption signal
carbon tax
capacity charge rates
flow charge rates
excess flow charge rates
last payments and compensations
```

Emitter 不观察：

```text
other emitters' requested capacities
other emitters' allocated capacities
other emitters' actions
```

Pipeline operator observation:

```text
total requested pipeline capacity
allocated/sold pipeline capacity
available pipeline capacity
pipeline utilization
contract nomination total
excess nomination total
unserved contract volume
revenue and liability last step
disruption state/forecast
```

Storage observation:

```text
own offered injection capacity
aggregate allocated storage capacity
pressure fraction
fill fraction
current injectivity
max injectable
contract delivered volume
excess delivered volume
actual injected
unserved contract volume
revenue and liability last step
disruption state/forecast
```

## 10. Required Outcome Fields

```text
allocated_pipeline_capacity[eid]
allocated_storage_capacity[eid, sid]

contract_pipeline_nomination[eid]
excess_pipeline_nomination[eid]
accepted_contract_pipeline[eid]
accepted_excess_pipeline[eid]
pipeline_unserved_contract[eid]

contract_storage_delivered[eid, sid]
excess_storage_delivered[eid, sid]
injected_contract[eid, sid]
injected_excess[eid, sid]
storage_unserved_contract[eid, sid]

pipeline_capacity_payment[eid]
storage_capacity_payment[eid, sid]
pipeline_flow_payment[eid]
storage_flow_payment[eid, sid]
pipeline_excess_payment[eid]
storage_excess_payment[eid, sid]

carbon_tax_payment[eid]
storage_credit_payment[eid]
pipeline_liability_payment[eid]
storage_liability_payment[eid, sid]
buffer_overflow_penalty[eid]
```

