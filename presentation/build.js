const pptxgen = require('pptxgenjs');
const html2pptx = require('/Users/guynachshon/.claude/plugins/cache/anthropic-agent-skills/document-skills/0f77e501e650/document-skills/pptx/scripts/html2pptx');
const path = require('path');

async function build() {
  const pptx = new pptxgen();
  pptx.layout = 'LAYOUT_16x9';
  pptx.author = 'Oz Labs Research';
  pptx.title = 'Learning RF Dynamics with JEPA World Models';

  const dir = path.join(__dirname, 'slides');

  const slides = [
    'slide01-title',
    'slide02-motivation',
    'slide03-task',
    'slide04-jepa',
    'slide05-architecture',
    'slide06-dataset',
    'slide07-collapse',
    'slide08-debugging',
    'slide09-more-fixes',
    'slide10-shortcuts',
    'slide11-v0-results',
    'slide12-v1-rollout',
    'slide13-v1-eval',
    'slide14-regimes',
    'slide15-embedding',
    'slide16-surprise',
    'slide17-imagination',
    'slide18-lessons',
    'slide19-future',
    'slide20-closing',
  ];

  for (const name of slides) {
    const htmlFile = path.join(dir, `${name}.html`);
    console.log(`Processing ${name}...`);
    try {
      const { slide, placeholders } = await html2pptx(htmlFile, pptx);

      // Add table to v0 results slide
      if (name === 'slide11-v0-results' && placeholders.length > 0) {
        slide.addTable([
          [
            { text: 'Method', options: { fill: { color: '1A1C22' }, color: 'E8520A', bold: true, fontSize: 9, fontFace: 'Courier New' } },
            { text: 'MSE', options: { fill: { color: '1A1C22' }, color: 'E8520A', bold: true, fontSize: 9, fontFace: 'Courier New' } },
            { text: 'vs Copy', options: { fill: { color: '1A1C22' }, color: 'E8520A', bold: true, fontSize: 9, fontFace: 'Courier New' } },
          ],
          [
            { text: 'RF-LeWM v0', options: { color: 'E8E9ED', fontSize: 9, fontFace: 'Courier New' } },
            { text: '1.469', options: { color: 'E8E9ED', bold: true, fontSize: 9, fontFace: 'Courier New' } },
            { text: '-33%', options: { color: '2A7A4B', bold: true, fontSize: 9, fontFace: 'Courier New' } },
          ],
          [
            { text: 'Copy-last', options: { color: '8B8D97', fontSize: 9, fontFace: 'Courier New' } },
            { text: '2.187', options: { color: '8B8D97', fontSize: 9, fontFace: 'Courier New' } },
            { text: 'baseline', options: { color: '55576A', fontSize: 9, fontFace: 'Courier New' } },
          ],
          [
            { text: 'Mean-context', options: { color: '8B8D97', fontSize: 9, fontFace: 'Courier New' } },
            { text: '1.481', options: { color: '8B8D97', fontSize: 9, fontFace: 'Courier New' } },
            { text: '-32%', options: { color: '8B8D97', fontSize: 9, fontFace: 'Courier New' } },
          ],
          [
            { text: 'Zero', options: { color: 'D4A017', fontSize: 9, fontFace: 'Courier New' } },
            { text: '1.163', options: { color: 'D4A017', bold: true, fontSize: 9, fontFace: 'Courier New' } },
            { text: '-47%', options: { color: 'D4A017', fontSize: 9, fontFace: 'Courier New' } },
          ],
        ], {
          ...placeholders[0],
          border: { pt: 0.5, color: '1E2028' },
          fill: { color: '13151A' },
          align: 'center',
          valign: 'middle',
          rowH: [0.3, 0.3, 0.3, 0.3, 0.3],
        });
      }

    } catch (e) {
      console.error(`FAILED: ${name} — ${e.message}`);
      process.exit(1);
    }
  }

  const outPath = path.join(__dirname, 'RF-LeWM-Research.pptx');
  await pptx.writeFile({ fileName: outPath });
  console.log(`\nPresentation saved to ${outPath}`);
}

build().catch(e => { console.error(e); process.exit(1); });
