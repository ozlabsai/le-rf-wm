const pptxgen = require('pptxgenjs');
const html2pptx = require('/Users/guynachshon/.claude/plugins/cache/anthropic-agent-skills/document-skills/0f77e501e650/document-skills/pptx/scripts/html2pptx');
const path = require('path');

async function build() {
  const pptx = new pptxgen();
  pptx.layout = 'LAYOUT_16x9';
  pptx.author = 'Oz Labs Research';
  pptx.title = 'RF-LeWM: A JEPA World Model for RF Spectral Environments';

  const dir = path.join(__dirname, 'slides');

  // Slide 1: Title
  await html2pptx(path.join(dir, 'slide01-title.html'), pptx);

  // Slide 2: Problem
  await html2pptx(path.join(dir, 'slide02-problem.html'), pptx);

  // Slide 3: Architecture
  await html2pptx(path.join(dir, 'slide03-architecture.html'), pptx);

  // Slide 4: Data
  await html2pptx(path.join(dir, 'slide04-data.html'), pptx);

  // Slide 5: Evolution
  await html2pptx(path.join(dir, 'slide05-evolution.html'), pptx);

  // Slide 6: Results (with table)
  const { slide: slide6, placeholders } = await html2pptx(path.join(dir, 'slide06-results.html'), pptx);
  if (placeholders.length > 0) {
    const tableData = [
      [
        { text: 'Metric', options: { fill: { color: '1A1C22' }, color: 'E8520A', bold: true, fontSize: 10, fontFace: 'Courier New' } },
        { text: 'RF-LeWM', options: { fill: { color: '1A1C22' }, color: 'E8520A', bold: true, fontSize: 10, fontFace: 'Courier New' } },
        { text: 'Copy-last', options: { fill: { color: '1A1C22' }, color: 'E8520A', bold: true, fontSize: 10, fontFace: 'Courier New' } },
        { text: 'Mean-ctx', options: { fill: { color: '1A1C22' }, color: 'E8520A', bold: true, fontSize: 10, fontFace: 'Courier New' } },
        { text: 'Zero', options: { fill: { color: '1A1C22' }, color: 'E8520A', bold: true, fontSize: 10, fontFace: 'Courier New' } },
        { text: 'vs Copy', options: { fill: { color: '1A1C22' }, color: 'E8520A', bold: true, fontSize: 10, fontFace: 'Courier New' } },
      ],
      [
        { text: '1-step MSE', options: { color: '8B8D97', fontSize: 10, fontFace: 'Courier New' } },
        { text: '1.270', options: { color: 'E8E9ED', bold: true, fontSize: 10, fontFace: 'Courier New' } },
        { text: '2.187', options: { color: '8B8D97', fontSize: 10, fontFace: 'Courier New' } },
        { text: '1.481', options: { color: '8B8D97', fontSize: 10, fontFace: 'Courier New' } },
        { text: '1.163', options: { color: '55576A', fontSize: 10, fontFace: 'Courier New' } },
        { text: '-42%', options: { color: '2A7A4B', bold: true, fontSize: 10, fontFace: 'Courier New' } },
      ],
      [
        { text: '12-step MSE', options: { color: '8B8D97', fontSize: 10, fontFace: 'Courier New' } },
        { text: '1.558', options: { color: 'E8E9ED', bold: true, fontSize: 10, fontFace: 'Courier New' } },
        { text: '2.233', options: { color: '8B8D97', fontSize: 10, fontFace: 'Courier New' } },
        { text: '1.564', options: { color: '8B8D97', fontSize: 10, fontFace: 'Courier New' } },
        { text: '1.176', options: { color: '55576A', fontSize: 10, fontFace: 'Courier New' } },
        { text: '-30%', options: { color: '2A7A4B', bold: true, fontSize: 10, fontFace: 'Courier New' } },
      ],
      [
        { text: 'Delta cosine', options: { color: '8B8D97', fontSize: 10, fontFace: 'Courier New' } },
        { text: '0.641', options: { color: 'E8E9ED', bold: true, fontSize: 10, fontFace: 'Courier New' } },
        { text: '0.000', options: { color: '8B8D97', fontSize: 10, fontFace: 'Courier New' } },
        { text: '0.559', options: { color: '8B8D97', fontSize: 10, fontFace: 'Courier New' } },
        { text: 'N/A', options: { color: '55576A', fontSize: 10, fontFace: 'Courier New' } },
        { text: '+15%', options: { color: '2A7A4B', bold: true, fontSize: 10, fontFace: 'Courier New' } },
      ],
    ];
    slide6.addTable(tableData, {
      ...placeholders[0],
      border: { pt: 0.5, color: '1E2028' },
      fill: { color: '13151A' },
      align: 'center',
      valign: 'middle',
      rowH: [0.35, 0.35, 0.35, 0.35],
    });
  }

  // Slide 7: Diagnostics
  await html2pptx(path.join(dir, 'slide07-diagnostics.html'), pptx);

  // Slide 8: MAE
  await html2pptx(path.join(dir, 'slide08-mae.html'), pptx);

  // Slide 9: Bridge
  await html2pptx(path.join(dir, 'slide09-bridge.html'), pptx);

  // Slide 10: Demo
  await html2pptx(path.join(dir, 'slide10-demo.html'), pptx);

  // Slide 11: Decisions
  await html2pptx(path.join(dir, 'slide11-decisions.html'), pptx);

  // Slide 12: Next steps
  await html2pptx(path.join(dir, 'slide12-next.html'), pptx);

  // Slide 13: Summary
  await html2pptx(path.join(dir, 'slide13-summary.html'), pptx);

  const outPath = path.join(__dirname, 'RF-LeWM-v1-Research.pptx');
  await pptx.writeFile({ fileName: outPath });
  console.log(`Presentation saved to ${outPath}`);
}

build().catch(e => { console.error(e); process.exit(1); });
