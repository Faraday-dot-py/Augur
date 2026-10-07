(function () {
  var script = document.currentScript;
  var model = (script && script.dataset.model) || 'token';
  var scatter = (script && script.dataset.scatter) || '../scatter/';
  var key = 'augur-notice-dismissed-' + model;

  try { if (localStorage.getItem(key) === '1') return; } catch (e) {}

  var COPY = {
    token: {
      summary: 'This model is not being developed further. See why, and what it is good for.',
      why: [
        'This model learns one force per ball, plus walls, from data. It was meant to copy any particle simulation. It works at small scale but did not hold up as the number of bodies grew.',
        '<ul>' +
        '<li>In a gravity test with 100 to 1000 bodies, its error after 20 steps was 2.14. Simply assuming every body keeps its velocity scored 0.95.</li>' +
        '<li>Its wall force is written around one fixed box, so it does not carry over to simulations without walls.</li>' +
        '<li>With a speed limit in the simulation, it still broke the limit in about a third of test scenes.</li>' +
        '<li>Several attempts to fix its ball tracking, each tested on its own, made no clear improvement.</li>' +
        '</ul>'
      ]
    },
    gravity: {
      summary: 'This model is not being developed further. See why, and what it is good for.',
      why: [
        'This model scales well: trained on 100 to 1000 bodies, it stays accurate past 10,000 (error after 20 steps 0.0054, energy within 0.01% of the truth). It does so by building in the answer.',
        '<ul>' +
        '<li>Every pair of bodies is forced to push or pull along the line between them, by an amount that depends only on distance. That is a rule we wrote in, not one the model learned.</li>' +
        '<li>So it cannot represent friction, uneven or one-way interactions, or energy moving between kinds (motion to heat). We want a model that can learn those.</li>' +
        '<li>It ignores bodies farther than a cutoff, while real gravity reaches everything. This was a large part of its drift over long runs.</li>' +
        '</ul>'
      ]
    }
  };
  var c = COPY[model] || COPY.token;

  var what =
    '<p>The goal is one method that can copy any particle simulation, grows to larger areas and more bodies, runs in time proportional to the number of bodies, and lets us read the simulation’s rules back out of the model. Neither this model nor its sibling met all of that.</p>' +
    '<p>We moved to the <a href="' + scatter + '">scatter-field model</a>. Bodies deposit their mass onto a grid, a small network turns that grid into a force field, and bodies read the field at their own position. Cost grows with bodies plus grid size, no body pairs are compared, and no force law is written in. It is not finished: it is trained on 10 to 100 bodies, and its error after 20 steps (0.0039) is close to the central-force model’s (0.0025).</p>';

  var uses =
    '<h3>What models like this can be used for</h3>' +
    '<ul>' +
    '<li>Stand-ins for slow simulations: once trained, a step costs the same however the original was computed.</li>' +
    '<li>Physics for games and animation that is learned from examples instead of coded.</li>' +
    '<li>Teaching and exploration: the weights here are small and run in your browser, so you can inspect and edit them.</li>' +
    '<li>Rule discovery: fitting a model to data from a system whose rules are unknown. This is a goal; it has not been tested.</li>' +
    '</ul>';

  var css =
    '.augnotice{--an-bg:#06080c;--an-glass:rgba(12,16,24,.92);--an-line:#1c2535;--an-line2:#2a3850;--an-fg:#e4eaf5;--an-dim:#8390a8;--an-accent:#6fc3ff;' +
    'position:fixed;top:0;left:0;right:0;z-index:100;background:var(--an-glass);border-bottom:1px solid var(--an-line);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);' +
    'color:var(--an-fg);font:12px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;overflow:auto;max-height:100vh}' +
    '.augnotice *{box-sizing:border-box}' +
    '.augnotice .an-bar{display:flex;align-items:stretch;min-height:30px}' +
    '.augnotice button{background:none;border:0;color:inherit;font:inherit;cursor:pointer;padding:0 12px}' +
    '.augnotice button:focus-visible,.augnotice a:focus-visible{outline:2px solid var(--an-accent);outline-offset:-2px}' +
    '.augnotice .an-tg{flex:1;min-width:0;display:flex;align-items:center;gap:8px;text-align:left;font-family:ui-monospace,SFMono-Regular,"JetBrains Mono",Menlo,Consolas,monospace;font-size:11px;color:var(--an-dim);padding:4px 12px}' +
    '.augnotice .an-tg:hover{color:var(--an-fg)}' +
    '.augnotice .an-tg span{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}' +
    '.augnotice.an-open .an-tg span{white-space:normal}' +
    '.augnotice .an-tg i{font-style:normal;transition:transform .15s}' +
    '.augnotice.an-open .an-tg i{transform:rotate(180deg)}' +
    '.augnotice .an-x{font-size:18px;line-height:1;color:var(--an-dim);min-width:44px}' +
    '.augnotice .an-x:hover{color:var(--an-fg)}' +
    '.augnotice .an-body{max-width:680px;padding:4px 16px 14px 12px;color:var(--an-dim)}' +
    '.augnotice .an-body[hidden]{display:none}' +
    '.augnotice p{margin:8px 0}' +
    '.augnotice ul{margin:6px 0;padding-left:18px}' +
    '.augnotice li{margin:4px 0}' +
    '.augnotice h3{margin:14px 0 4px;font-size:12px;font-weight:600;color:var(--an-fg)}' +
    '.augnotice a{color:var(--an-accent)}' +
    'body.has-augnotice header,body.has-augnotice aside{margin-top:30px}' +
    '@media (prefers-reduced-motion:reduce){.augnotice .an-tg i{transition:none}}';

  var style = document.createElement('style');
  style.textContent = css;
  document.head.appendChild(style);

  var el = document.createElement('div');
  el.className = 'augnotice';
  el.setAttribute('role', 'region');
  el.setAttribute('aria-label', 'Notice about this model');
  el.innerHTML =
    '<div class="an-bar">' +
    '<button type="button" class="an-tg" aria-expanded="false" aria-controls="an-body-' + model + '"><span>' + c.summary + '</span><i aria-hidden="true">▾</i></button>' +
    '<button type="button" class="an-x" aria-label="Dismiss this notice">×</button>' +
    '</div>' +
    '<div class="an-body" id="an-body-' + model + '" hidden>' +
    '<h3>Why we moved on</h3>' + c.why.join('') + what + uses +
    '</div>';

  var tg = el.querySelector('.an-tg');
  var body = el.querySelector('.an-body');
  tg.addEventListener('click', function () {
    var open = body.hidden;
    body.hidden = !open;
    tg.setAttribute('aria-expanded', String(open));
    el.classList.toggle('an-open', open);
  });
  el.querySelector('.an-x').addEventListener('click', function () {
    el.remove();
    document.body.classList.remove('has-augnotice');
    try { localStorage.setItem(key, '1'); } catch (e) {}
  });

  document.body.classList.add('has-augnotice');
  document.body.appendChild(el);
})();
