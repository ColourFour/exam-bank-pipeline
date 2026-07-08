# Topic Packet Visual Bug Registry

- Complete: `True`
- Bug records: `2`
- Seed bugs tracked: `0`

| Bug | Status | Resolution | Categories | Packet | Page | Root cause |
| --- | --- | --- | --- | --- | ---: | --- |
| `tpva_p1_functions_page_0044` | bug | fixed | question_crop | `p1_functions` | 44 | 11summer15_q08 source crop previously included the opening line of the following question; foreign-question boundary trimming removed the spillover and the rebuilt packet page now ends at the intended Q8 content. |
| `tpva_p1_functions_page_0060` | bug | fixed | question_crop | `p1_functions` | 60 | 13summer11_q10 source crop previously included the tail of the previous question; top-boundary trimming removed the spillover and the rebuilt packet page now starts at Q10 with all current content intact. |
