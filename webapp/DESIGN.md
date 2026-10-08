# Drug Review workspace

The audience is a student presenting a review-text ML project. The main action is
submitting a review and understanding its predicted experience class and rating.

- Colors: paper white `#ffffff`, mist blue `#edf3f8`, ink navy `#172d47`,
  marine blue `#315c85`, teal `#267a69`, amber `#9c6c18`.
- Type: Trebuchet MS for conversational headings; Segoe UI for forms and data.
- Layout: a quiet left navigation rail, a large review editor, and an adjacent
  prediction panel. Model comparison uses a wide table and labeled score bars.
- Alignment: left-aligned copy, consistent form labels, right-aligned numeric data.
- Subject-specific focus: the submitted review remains visible beside its output.
  Drug and condition are context fields; the page explains that only text is modeled.
  Predictions are patient-experience labels, not treatment recommendations.

```text
Navigation | Title + connection status
           | Review editor          | Prediction
           | Genuine sample choices | Class / rating / model score
           | What the model uses    | Explanation
           | Results view: real validation comparison and evaluation notes
```

The initial generic dashboard-card idea was replaced with an editor workspace.
The review is the central object, not decorative metrics. The results view uses
real experiment records; no prefilled predictions or simulated loading scores.
Keyboard focus, mobile stacking, reduced motion, and readable contrasts are required.
