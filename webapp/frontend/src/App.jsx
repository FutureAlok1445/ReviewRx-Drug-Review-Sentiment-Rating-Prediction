import { useEffect, useState } from 'react';

async function api(path, options = {}) {
  const response = await fetch('/api' + path, options);
  const content = await response.json();
  if (!response.ok) {
    const detail = typeof content.detail === 'string' ? content.detail : 'Check the review and try again.';
    throw new Error(detail);
  }
  return content;
}

const percent = (value) => Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : '—';
const decimal = (value) => Number.isFinite(value) ? value.toFixed(4) : '—';

function App() {
  const [view, setView] = useState('analyze');
  const [connection, setConnection] = useState('connecting');
  const [examples, setExamples] = useState([]);
  const [results, setResults] = useState(null);
  const [form, setForm] = useState({ drug: '', condition: '', review: '' });
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [sample, setSample] = useState(null);

  useEffect(() => {
    let active = true;
    Promise.all([api('/health'), api('/examples'), api('/results')])
      .then(([, sampleData, resultData]) => {
        if (active) { setConnection('ready'); setExamples(sampleData); setResults(resultData); }
      })
      .catch(() => { if (active) setConnection('offline'); });
    return () => { active = false; };
  }, []);

  const update = (event) => {
    setForm((current) => ({ ...current, [event.target.name]: event.target.value }));
    setPrediction(null); setSample(null); setError('');
  };

  const loadExample = (example) => {
    setForm({ drug: example.drug, condition: example.condition, review: example.review });
    setSample(example); setPrediction(null); setError('');
  };

  const analyze = async (event) => {
    event.preventDefault();
    if (!form.review.trim()) { setError('Enter a review before analyzing.'); return; }
    setBusy(true); setError(''); setPrediction(null);
    try {
      const answer = await api('/predict', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(form),
      });
      setPrediction(answer); setConnection('ready');
    } catch (issue) {
      setError(issue instanceof TypeError ? 'Cannot reach the backend. Start FastAPI, then try again.' : issue.message);
    } finally { setBusy(false); }
  };

  const experience = prediction?.predicted_experience_class;
  const experienceDescription = {
    Positive: 'The text is associated with the higher rating group (7–10).',
    Neutral: 'The text is associated with the middle rating group (4–6).',
    Negative: 'The text is associated with the lower rating group (1–3).',
  };

  return (
    <div className="workspace">
      <aside className="sidebar">
        <a className="brand" href="#" onClick={(event) => { event.preventDefault(); setView('analyze'); }}>
          <span className="brand-symbol" aria-hidden="true">✚</span>
          <span>Drug Review<span className="brand-subtitle">Analysis workspace</span></span>
        </a>
        <nav aria-label="Main navigation">
          <button className={view === 'analyze' ? 'nav-button active' : 'nav-button'} onClick={() => setView('analyze')}>Analyze a review</button>
          <button className={view === 'results' ? 'nav-button active' : 'nav-button'} onClick={() => setView('results')}>Model results</button>
        </nav>
        <a className="notebook-download" href="/api/notebook" download>Download training notebook</a>
        <div className="sidebar-note">
          <strong>A college ML project</strong>
          <p>Patient-reported experiences, made easier to explore.</p>
          <span>Research & education</span>
        </div>
      </aside>

      <main>
        <header className="topbar">
          <span>Review-text research</span>
          <span className={`connection ${connection}`}><span className="status-dot" />
            {connection === 'ready' ? 'Model connected' : connection === 'offline' ? 'Backend unavailable' : 'Connecting to model'}
          </span>
        </header>
        {view === 'analyze' ? (
          <>
            <div className="page-heading"><h1>Read the experience<br className="desktop-break" /> behind the review.</h1>
              <p>Explore a review’s predicted patient experience and rating. Start with your own text or a genuine dataset example.</p>
            </div>
            <div className="analysis-layout">
              <section className="editor-panel" aria-labelledby="review-heading">
                <div className="panel-heading"><h2 id="review-heading">Your review</h2><span>Text-based analysis</span></div>
                <form onSubmit={analyze}>
                  <div className="context-fields">
                    <label>Drug name <span>(optional)</span><input name="drug" value={form.drug} onChange={update} maxLength={200} placeholder="Add drug context" disabled={busy} /></label>
                    <label>Condition <span>(optional)</span><input name="condition" value={form.condition} onChange={update} maxLength={200} placeholder="Add condition context" disabled={busy} /></label>
                  </div>
                  <label className="review-label">Patient review
                    <textarea name="review" value={form.review} onChange={update} maxLength={20000} rows={10} required disabled={busy}
                      placeholder="Paste the patient’s review here. Include their own description of benefits, side effects, and overall experience." aria-describedby="review-help" />
                  </label>
                  <div className="editor-footer"><p id="review-help">Only review text influences predictions.</p><span>{form.review.length.toLocaleString()} / 20,000</span></div>
                  {error && <p className="error" role="alert">{error}</p>}
                  <button className="primary-button" type="submit" disabled={busy}>{busy ? 'Analyzing review…' : 'Analyze review'}<span aria-hidden="true">↗</span></button>
                </form>
                <div className="examples"><h3>Try a genuine dataset example</h3>
                  <div className="example-buttons">{examples.map((example) => <button key={example.uniqueID} disabled={busy} onClick={() => loadExample(example)}>{example.label} review</button>)}</div>
                  {sample && <p className="sample-note">Dataset record {sample.uniqueID}. Its recorded rating is {sample.true_rating}/10; the prediction may differ.</p>}
                </div>
              </section>

              <section className={`prediction-panel ${experience?.toLowerCase() || ''}`} aria-labelledby="prediction-heading" aria-live="polite" aria-busy={busy}>
                <h2 id="prediction-heading">Prediction</h2>
                {prediction ? <>
                  <span className="experience-tag">{experience} experience</span>
                  <p className="experience-description">{experienceDescription[experience]}</p>
                  <div className="rating-output"><span className="output-label">Predicted rating</span><div><strong>{prediction.predicted_rating}</strong><span> / 10</span></div></div>
                  <div className="rating-scale" aria-label={`Rating ${prediction.predicted_rating} out of 10`}>{Array.from({length:10},(_,i) => <span key={i} className={i < prediction.predicted_rating ? 'filled' : ''} />)}</div>
                  <div className="confidence"><div><span>Model score</span><strong>{percent(prediction.confidence)}</strong></div><div className="confidence-track"><span style={{width:`${prediction.confidence*100}%`}} /></div>
                    <p>Maximum class probability. This score is not a verified probability of correctness.</p></div>
                  <p className="prediction-context">{prediction.drug || 'Drug not specified'}<br />{prediction.condition || 'Condition not specified'}</p>
                </> : <div className="empty-prediction"><div className="review-glyph" aria-hidden="true"><span /><span /><span /></div><h3>{busy ? 'Reading your review…' : 'Your analysis starts here'}</h3><p>{busy ? 'The trained model is processing the review text.' : 'Enter a review and select Analyze review to see the model’s output.'}</p></div>}
                <div className="model-caption"><span>Experience model</span><strong>{results?.sentiment_selection || 'Waiting for backend'}</strong></div>
              </section>
            </div>
            <div className="interpretation-note"><strong>How to read this result</strong><p>Experience classes come from patient ratings, not independent clinical labels. Sentiment and predicted rating are separate outputs and can disagree. Use this analysis for learning; it does not assess treatment safety or recommend a medicine.</p></div>
          </>
        ) : <ResultsView results={results} />}
        <footer>Drug Review Analysis<span>Built from verified project experiments.</span></footer>
      </main>
    </div>
  );
}

function ResultsView({ results }) {
  if (!results) return <div className="page-heading"><h1>Model results</h1><p>Start the backend to load verified experiment results.</p></div>;
  const models = [...results.sentiment_models].sort((a,b) => b.F1-a.F1);
  const currentTest = results.test_metrics.experience || results.test_metrics.selected_experience;
  const currentRating = results.test_metrics.rating || results.test_metrics.selected_rating;
  return <>
    <div className="page-heading"><h1>Evidence behind<br /> the prediction.</h1><p>Real model comparisons from the same validation records. Higher macro-F1 gives each experience class equal importance.</p></div>
    <div className="selection-strip"><div><span>Selected experience model</span><strong>{results.sentiment_selection}</strong></div><div><span>Selected rating model</span><strong>{results.rating_selection}</strong></div></div>
    <section className="results-panel"><h2>Experience model comparison</h2><p>CV uses three grouped training folds. Parameter and ensemble searches reuse these folds, so CV scores include selection optimism.</p>
      <div className="table-scroll"><table><thead><tr><th>Model</th><th>CV F1</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>Macro-F1</th><th>ROC-AUC</th></tr></thead><tbody>{models.map((row) => <tr key={row.Model} className={row.Model === results.sentiment_selection ? 'selected-row' : ''}><th scope="row">{row.Model}{row.Model === results.sentiment_selection && <span className="selected-label">Selected</span>}</th><td>{decimal(row['CV Score'])}</td><td>{percent(row.Accuracy)}</td><td>{decimal(row.Precision)}</td><td>{decimal(row.Recall)}</td><td>{decimal(row.F1)}</td><td>{decimal(row['ROC-AUC'])}</td></tr>)}</tbody></table></div>
    </section>
    <div className="results-grid"><section className="results-panel"><h2>Validation macro-F1</h2><div className="score-chart">{models.map((row) => <div className="score-row" key={row.Model}><div><span>{row.Model}</span><strong>{decimal(row.F1)}</strong></div><div className="score-track"><span style={{width:`${row.F1*100}%`}} /></div></div>)}</div></section>
      <section className="results-panel"><h2>Previously evaluated test</h2><p>{results.test_note}</p><dl className="test-metrics"><div><dt>Experience accuracy</dt><dd>{percent(currentTest?.Accuracy)}</dd></div><div><dt>Experience macro-F1</dt><dd>{decimal(currentTest?.F1)}</dd></div><div><dt>Rating MAE</dt><dd>{decimal(currentRating?.MAE)}</dd></div><div><dt>Rating within one point</dt><dd>{percent(currentRating?.['Within-1 accuracy'])}</dd></div></dl><p className="evaluation-note">Model selection uses training CV and validation. A new external test is needed for independent confirmation after these additional experiments.</p></section></div>
    <section className="results-panel"><h2>Rating model comparison</h2><p>MAE measures average error in rating points; lower is better.</p><div className="table-scroll"><table><thead><tr><th>Model</th><th>CV MAE</th><th>Validation MAE</th><th>RMSE</th><th>Exact accuracy</th><th>Within one point</th></tr></thead><tbody>{results.rating_models.map((row) => <tr key={row.Model}><th scope="row">{row.Model}</th><td>{decimal(row['CV MAE'])}</td><td>{decimal(row.MAE)}</td><td>{decimal(row.RMSE)}</td><td>{percent(row['Exact accuracy'])}</td><td>{percent(row['Within-1 accuracy'])}</td></tr>)}</tbody></table></div></section>
  </>;
}

export default App;
