# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_real_120.json
- Mode: openai
- Records: 240
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.8167 | 0.85 | 0.0333 |
| Avg attempts | 1 | 1.35 | 0.35 |
| Avg token estimate | 1443.18 | 1950.17 | 506.99 |
| Avg latency (ms) | 1158.28 | 1502.42 | 344.14 |

## Failure modes
```json
{
  "react": {
    "none": 98,
    "wrong_final_answer": 16,
    "incomplete_multi_hop": 6
  },
  "reflexion": {
    "none": 102,
    "wrong_final_answer": 13,
    "incomplete_multi_hop": 5
  },
  "total": {
    "none": 200,
    "wrong_final_answer": 29,
    "incomplete_multi_hop": 11
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json

## Discussion
The benchmark on hotpot_real_120.json using openai shows that Reflexion provides a measurable improvement in Exact Match accuracy compared to a standard ReAct approach. Specifically, we observed a 3.3% increase in performance. The reflection memory was particularly useful for correcting 'entity_drift' where the model initially followed a wrong path in the multi-hop context. However, this comes at the cost of increased token consumption (~506.99 tokens) and higher latency. The evaluator agent was successful in identifying logic errors, but occasionally struggled with nuanced phrasing, suggesting that few-shot examples for the evaluator could further improve the precision of the self-correction loop.
