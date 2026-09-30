/* Offline view: no requests, no training controls, no external dependencies. */
const finite = value => typeof value === 'number' && Number.isFinite(value);
const average = values => values.length ? values.reduce((a,b)=>a+b,0)/values.length : null;
const deviation = values => values.length > 1 ? Math.sqrt(values.reduce((s,x)=>s+(x-average(values))**2,0)/(values.length-1)) : null;
function filteredRuns(runs, filters) {
  return runs.filter(r => Object.entries(filters).every(([key,value]) => {
    if (!value) return true;
    if (key === 'search') return [r.job,r.family,r.variant,r.dynamics,r.campaignLabel].join(' ').toLowerCase().includes(value.toLowerCase());
    if (key === 'status' && value === 'data') return r.history.length > 0;
    return String(r[key]) === value;
  }));
}
function seedGroups(runs) {
  const groups = new Map();
  for (const r of runs.filter(r=>r.status==='complete' && finite(r.bestVal))) {
    const s=r.specs;
    const key=JSON.stringify([r.campaign,r.dynamics,r.family,r.variant,r.timesteps,r.stage,r.method,r.regularization,
      s.batch_size,s.learning_rate,s.epochs,s.bptt_window,s.early_stopping_patience,s.lr_reduce_patience,s.lr_reduce_factor,
      s.gradient_clip_norm,s.freeze_readout_epochs,s.minimum_finetune_epochs,s.hh_backend,s.hh_surrogate_width_mv,
      r.trainSamples,r.valSamples]);
    if (!groups.has(key)) groups.set(key,[]);
    groups.get(key).push(r);
  }
  return [...groups.values()].map(runs=>({runs,mean:average(runs.map(r=>r.bestVal)),sd:deviation(runs.map(r=>r.bestVal))})).sort((a,b)=>b.mean-a.mean);
}
function plotPoints(run, metric, mode, lower, upper) {
  return run.history.filter(r=>r.epoch>=lower && r.epoch<=upper).map(r=>({
    x:mode==='time' ? (finite(r.cumulativeSeconds) ? r.cumulativeSeconds/60 : null) : r.epoch,
    y:r[metric],row:r
  }));
}
if (typeof module !== 'undefined' && module.exports) module.exports={filteredRuns,seedGroups,plotPoints,average,deviation};
if (typeof document !== 'undefined') (()=>{
  const D=JSON.parse(document.getElementById('dataset').textContent), $=id=>document.getElementById(id);
  const escape=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const format=(x,d=2)=>finite(x)?x.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d}):'—';
  const pct=x=>finite(x)?format(x*100)+' %':'—';
  const names={complete:'Completed',running:'Running',pending:'Planned',interrupted:'Interrupted'};
  const colors=Array.from({length:6},(_,i)=>`var(--series-${i+1})`);
  const byId=new Map(D.runs.map(r=>[r.id,r]));
  let selected=[],tab='curves';
  selected=['lif__deterministic-rate__scale1__s14','izh__rate__scale1__s14','hh__deterministic-rate__scale0p25__s14']
    .map(job=>D.runs.find(r=>r.job===job)?.id).filter(Boolean);
  const filterIds=['dynamics','family','campaign','timesteps','seed','stage','status','search'];
  for(const key of ['dynamics','family','campaign','timesteps','seed']){
    const values=[...new Set(D.runs.map(r=>r[key]))].filter(v=>v!==null).sort((a,b)=>typeof a==='number'?a-b:String(a).localeCompare(String(b)));
    for(const value of values){const option=document.createElement('option');option.value=value;option.textContent=key==='campaign'?D.runs.find(r=>r.campaign===value).campaignLabel:value;$(key).append(option);}
  }
  const stamp=new Date(D.builtUtc).toLocaleString('en-US',{dateStyle:'medium',timeStyle:'short'});
  $('datestamp').textContent='HTML BUILT · '+stamp;
  const recorded=D.runs.filter(r=>r.history.length), timed=recorded.filter(r=>r.history.some(h=>finite(h.epochSeconds)));
  $('summary').innerHTML=[[recorded.length,'Stages with learning curves'],[recorded.reduce((n,r)=>n+r.history.filter(h=>h.epoch>0).length,0),'Training epochs'],[timed.length,'Stages with timing data'],[D.runs.filter(r=>r.status==='pending').length,'Still planned']].map(([n,label])=>`<div><strong>${n.toLocaleString('en-US')}</strong><span>${label}</span></div>`).join('');
  const live=D.liveStatus;
  const importStamp=D.importedUtc?new Date(D.importedUtc).toLocaleString('en-US',{dateStyle:'medium',timeStyle:'short'}):'not documented';
  $('snapshot').textContent='Completed campaign: 66 fits · 50 epochs plus baseline · seed 14 · T = 8. Offline snapshot. Test values are final endpoints; a single seed cannot estimate variability.';
  function runs(){return selected.map(id=>byId.get(id)).filter(Boolean);}
  function filtered(){return filteredRuns(D.runs,Object.fromEntries(filterIds.map(id=>[id,$(id).value])));}
  function message(text=''){$('message').textContent=text;}
  function toggle(id){if(selected.includes(id))selected=selected.filter(x=>x!==id);else if(selected.length<6)selected.push(id);else {message('Select up to six stages at a time to keep curves readable.');return;}message();render();}
  function table(headers,rows){return `<table><thead><tr>${headers.map(h=>`<th>${h}</th>`).join('')}</tr></thead><tbody>${rows.join('')||`<tr><td colspan="${headers.length}" class="empty">No matching data.</td></tr>`}</tbody></table>`;}
  const campaignShort={full_encoding_50:'50 epochs'};
  function rowTitle(r){return `${r.dynamics} · ${r.family} · ${r.variant} · T${r.timesteps} / s${r.seed} · ${r.stage} · ${campaignShort[r.campaign]}`;}
  function drawLegend(){$('legend').innerHTML=selected.map((id,i)=>`<button data-remove="${escape(id)}" title="Remove from comparison"><i class="swatch" style="background:${colors[i]}"></i>${escape(rowTitle(byId.get(id)))} <span aria-hidden="true">×</span></button>`).join('')||'<span class="status-text">Select training stages below to compare.</span>';document.querySelectorAll('[data-remove]').forEach(el=>el.onclick=()=>toggle(el.dataset.remove));}
  function experimentTable(){
    const list=filtered(),sort=$('sort').value;
    list.sort((a,b)=>sort==='validation'?(b.bestVal??-1)-(a.bestVal??-1):sort==='recent'?Number(b.campaign==='observability')-Number(a.campaign==='observability')||a.job.localeCompare(b.job):rowTitle(a).localeCompare(rowTitle(b)));
    $('resultCount').textContent=`${list.length} stages match the filters · ${selected.length} in comparison (selection is retained when filtering)`;
    $('experiments').innerHTML=table(['Compare','Model / encoding','Campaign','Seed / T','Best val.','Best / last epoch','Time / epoch¹','Status'],list.map(r=>{
      const times=r.history.map(h=>h.epochSeconds).filter(finite);
      return `<tr class="${selected.includes(r.id)?'selected':''}"><td><input type="checkbox" data-select="${escape(r.id)}" ${selected.includes(r.id)?'checked':''} aria-label="${escape(rowTitle(r))}: select"></td><td><div class="name">${escape(r.dynamics)} · ${escape(r.family)}</div><div class="small">${escape(r.variant)} · ${escape(r.method)} · ${escape(r.stage)}</div></td><td>${escape(r.campaignLabel)}</td><td>${r.seed} / ${r.timesteps}</td><td class="number">${pct(r.bestVal)}</td><td class="number">${r.bestEpoch??'—'} / ${r.lastEpoch??'—'}</td><td class="number">${times.length?format(average(times),1)+' s':'—'}</td><td><span class="tag ${r.status}">${names[r.status]}</span></td></tr>`;
    }));
    document.querySelectorAll('[data-select]').forEach(el=>el.onchange=()=>{toggle(el.dataset.select);el.checked=selected.includes(el.dataset.select);});
  }
  const labels={valAcc:'Validation',trainAcc:'Train · Checkpoint',onlineAcc:'Train · online',trainLoss:'Training objective',trainCe:'Train · Online-CE',trainEvalCe:'Train · Checkpoint-CE',valCe:'Validation · CE',trainReg:'Weighted regularization',lr:'Learning rate',ties:'Tie-Rate',silent:'No prediction',hiddenSpikes:'Hidden spikes / image',outputSpikes:'Output spikes / image',grad0:'Gradient L0',grad1:'Gradient L1',changed0:'Changed INT8 weights L0',changed1:'Changed INT8 weights L1',lutClamps:'LUT-Clamps',epochSeconds:'Whole epoch',trainSeconds:'Training',valSeconds:'Validation',trainEvalSeconds:'Training evaluation',saveSeconds:'Saving'};
  const percentage=new Set(['valAcc','trainAcc','onlineAcc','ties','silent']);
  function chart(id,metrics,unit){
    const host=$(id),mode=$('xmode').value,lower=Math.max(0,Number($('epochMin').value)||0),upper=$('epochMax').value?Number($('epochMax').value):Infinity;
    const series=runs().flatMap((run,index)=>metrics.map((metric,j)=>({run,metric,color:colors[index],dash:j===1,points:plotPoints(run,metric,mode,lower,upper)})));
    const valid=series.flatMap(s=>s.points.filter(p=>finite(p.x)&&finite(p.y)));
    if(!valid.length){host.innerHTML='<div class="plot-empty">'+(selected.length?'No measurements of this metric are available for the selection.':'Select training stages in the table.')+'</div>';return;}
    const width=Math.max(host.clientWidth-20,260),height=host.clientHeight-4,left=58,right=19,top=16,bottom=44,pw=width-left-right,ph=height-top-bottom;
    let xmin=Math.min(...valid.map(p=>p.x)),xmax=Math.max(...valid.map(p=>p.x));if(xmax===xmin)xmax=xmin+1;
    const isPct=metrics.every(m=>percentage.has(m)),scale=isPct?100:1;
    const ymin=0,ymax=isPct?100:Math.max(...valid.map(p=>p.y))*1.12||1;
    const x=v=>left+(v-xmin)/(xmax-xmin)*pw,y=v=>top+ph-(v*scale-ymin)/(ymax-ymin)*ph;
    const nice=v=>isPct?format(v,0)+' %':v>999?format(v/1000,1)+'k':v<0.01&&v>0?v.toExponential(1):format(v,v<1?3:1);
    let svg=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escape(metrics.map(m=>labels[m]).join(', '))}"><text x="${left}" y="10" fill="var(--axis)" font-size="9">${isPct?'%':unit==='s'?'Seconds':''}</text>`;
    for(let i=0;i<=4;i++){const yy=top+ph-i*ph/4;svg+=`<line x1="${left}" y1="${yy}" x2="${width-right}" y2="${yy}" stroke="var(--grid)"/><text x="${left-9}" y="${yy+4}" text-anchor="end" fill="var(--axis)" font-size="10">${nice(ymax*i/4)}</text>`;}
    const ticks=width<380?3:5,seen=new Set();
    for(let i=0;i<ticks;i++){const raw=xmin+(xmax-xmin)*i/(ticks-1),v=mode==='epoch'?Math.round(raw):raw;if(seen.has(v))continue;seen.add(v);svg+=`<text x="${x(v)}" y="${height-bottom+20}" text-anchor="middle" fill="var(--axis)" font-size="10">${format(v,mode==='epoch'?0:1)}</text>`;}
    svg+=`<text x="${left+pw/2}" y="${height-5}" text-anchor="middle" fill="var(--axis)" font-size="10">${mode==='epoch'?'Training stage epoch':'Cumulative epoch time (min)'}</text>`;
    for(const s of series){let path='',pen=false;for(const p of s.points){if(!finite(p.x)||!finite(p.y)){pen=false;continue;}path+=(pen?'L':'M')+x(p.x).toFixed(2)+','+y(p.y).toFixed(2);pen=true;}
      svg+=`<path d="${path}" stroke="${s.color}" fill="none" stroke-width="2" ${s.dash?'stroke-dasharray="5 4"':''}/>`;
      for(const p of s.points.filter(p=>finite(p.x)&&finite(p.y))){const best=p.row.epoch===s.run.bestEpoch;svg+=`<circle cx="${x(p.x)}" cy="${y(p.y)}" r="${best?4.5:2}" fill="${best?'var(--surface)':s.color}" stroke="${s.color}" stroke-width="${best?2:0}"/>`;if(p.row.lrReduced)svg+=`<path d="M${x(p.x)},${y(p.y)-7}l-4,-6h8z" fill="${s.color}"/>`;}
    }
    svg+=`<line data-guide x1="0" x2="0" y1="${top}" y2="${top+ph}" stroke="var(--guide)" stroke-dasharray="3 3" visibility="hidden"/><rect data-hover x="${left}" y="${top}" width="${pw}" height="${ph}" fill="transparent"/></svg>`;
    host.innerHTML=svg;const overlay=host.querySelector('[data-hover]'),guide=host.querySelector('[data-guide]');
    overlay.onpointermove=e=>{const rect=host.querySelector('svg').getBoundingClientRect(),px=(e.clientX-rect.left)*width/rect.width,target=xmin+(px-left)/pw*(xmax-xmin);let content='';let first=true;
      for(const s of series){const points=s.points.filter(p=>finite(p.x)&&finite(p.y));if(!points.length)continue;const p=points.reduce((a,b)=>Math.abs(a.x-target)<Math.abs(b.x-target)?a:b);if(first){guide.setAttribute('x1',x(p.x));guide.setAttribute('x2',x(p.x));guide.setAttribute('visibility','visible');first=false;}content+=`<div><span style="color:${s.color}">●</span> ${escape(s.run.dynamics+' · '+s.run.variant+' · s'+s.run.seed)} · E${p.row.epoch}<br>${escape(labels[s.metric])}: <b>${percentage.has(s.metric)?pct(p.y):format(p.y,p.y<.01?6:3)}${unit?' '+unit:''}</b></div>`;}
      const tip=$('tooltip');tip.innerHTML=content;tip.style.display='block';tip.style.left=Math.max(8,Math.min(e.clientX+14,innerWidth-tip.offsetWidth-12))+'px';tip.style.top=Math.max(8,Math.min(e.clientY+14,innerHeight-tip.offsetHeight-12))+'px';};
    overlay.onpointerleave=()=>{$('tooltip').style.display='none';guide.setAttribute('visibility','hidden');};
  }
  function drawCharts(){
    if(tab==='curves'){
      const a=$('accuracyMetric').value;chart('accuracyChart',a==='both'?['valAcc','trainAcc']:[a]);
      const l=$('lossMetric').value,regs=new Set(runs().map(r=>r.regularization));
      $('lossNote').textContent=l==='trainLoss'?'Online objective: cross-entropy plus weighted regularization, if applicable. Different regularization objectives are not overlaid.':'Cross-entropy is shown separately from the regularized training objective. Dashed: training split at the checkpoint.';
      if(l==='trainLoss'&&regs.size>1)$('lossChart').innerHTML='<div class="plot-empty">The selection contains different regularization objectives. Select matching stages or switch to cross-entropy.</div>';
      else chart('lossChart',l==='ce'?['valCe','trainEvalCe']:[l]);
    }
    if(tab==='diagnostics'){chart('diagnosticChart',[$('diagnosticMetric').value]);chart('timeChart',[$('timeMetric').value],'s');}
  }
  function endpoints(){
    $('endpoints').innerHTML=table(['Training stage','Checkpoint','Validation','Test','Test-CE','Last val.','Recorded epoch time'],runs().map(r=>`<tr><td>${escape(rowTitle(r))}</td><td>E${r.bestEpoch??'—'}</td><td class="number">${pct(r.bestVal)}</td><td class="number">${pct(r.test?.accuracy)}</td><td class="number">${format(r.test?.cross_entropy,4)}</td><td class="number">${pct(r.history.at(-1)?.valAcc)}</td><td class="number">${finite(r.history.at(-1)?.cumulativeSeconds)?format(r.history.at(-1).cumulativeSeconds/60,1)+' min':'—'}</td></tr>`));
  }
  function hints(){
    $('hints').innerHTML=runs().map(r=>{const points=r.history.filter(h=>finite(h.valAcc)&&h.epoch>0),last=points.at(-1),earlier=points.length>=6?points.at(-6):null;let text=[];
      if(last&&earlier)text.push(`Validation change over the last 5 epochs: ${format((last.valAcc-earlier.valAcc)*100)} percentage points.`);
      if(r.bestEpoch===0)text.push('The baseline remains the best checkpoint.');
      if(last&&r.status==='complete'&&last.epoch>=r.specs.epochs&&r.bestEpoch>=last.epoch-1)text.push('Best checkpoint near the epoch limit: consider a targeted review of the training budget.');
      if(r.status==='running')text.push('Run incomplete; no final conclusion about convergence yet.');
      if(!r.history.some(h=>finite(h.trainAcc)))text.push('Training-split accuracy is missing. The training/validation gap cannot be measured.');
      return `<div class="hint"><b>${escape(rowTitle(r))}</b><br>${escape(text.join(' ')||'No automatic interpretation. Assess individual metrics together with the training parameters.')}</div>`;}).join('');
  }
  function specs(){
    $('specs').innerHTML=runs().map((r,i)=>{
      const s=r.specs,fields=[['Campaign',r.campaignLabel],['Training method',r.method+' · '+r.stage],['Neurons',r.dynamics],['Topology',Object.values(r.topology).length?`${r.topology.input_neurons} → ${r.topology.hidden_neurons} → ${r.topology.output_neurons}`:'—'],['Encoding',r.family+' · '+r.variant],['T',r.timesteps],['Training seed',r.seed],['Training / Validation',`${r.trainSamples??'?'} / ${r.valSamples??'?'}`],['Batch',s.batch_size],['Initial LR',s.learning_rate],['Epoch limit',s.epochs],['BPTT window',s.bptt_window===0?'full':s.bptt_window],['Regularization λ',r.regularization],['LR-Patience',s.lr_reduce_patience],['Early-Stop-Patience',s.early_stopping_patience],['Gradient-Clipping',s.gradient_clip_norm],['HH-Surrogate (mV)',s.hh_surrogate_width_mv],['HH-Backend',s.hh_backend],['Best / last epoch',`${r.bestEpoch??'—'} / ${r.lastEpoch??'—'}`]];
      return `<article class="panel spec"><h3><i class="swatch" style="background:${colors[i]}"></i> ${escape(r.dynamics+' · '+r.variant+' · s'+r.seed)}</h3><p>${escape(r.job)}</p><dl>${fields.map(([k,v])=>`<dt>${escape(k)}</dt><dd>${escape(v??'not recorded')}</dd>`).join('')}</dl>${r.parent?`<p>Resumed from ${escape(r.parent)}. The selected parent checkpoint is the starting point.</p>${byId.has(r.parent)?`<button data-parent="${escape(r.parent)}">Add base to comparison</button>`:''}`:''}<div class="source">${escape(r.source)}<br>SHA-256: ${escape(r.sha256??'Individual files: see download manifest')}</div><details><summary>Show original parameters</summary><pre>${escape(JSON.stringify(s,null,2))}</pre></details></article>`;
    }).join('')||'<div class="empty">Select training stages in the table.</div>';
    document.querySelectorAll('[data-parent]').forEach(el=>el.onclick=()=>{if(!selected.includes(el.dataset.parent))toggle(el.dataset.parent);});
  }
  function seeds(){
    const groups=seedGroups(filtered());
    $('seedTable').innerHTML=table(['Neurons / encoding','Campaign / stage','Seeds (n)','Validation ± SD','Test ± SD'],groups.map(g=>{
      const r=g.runs[0],tests=g.runs.map(r=>r.test?.accuracy).filter(finite);
      const spread=(mean,sd)=>pct(mean)+(finite(sd)?' ± '+format(sd*100)+' pp':' · SD —');
      return `<tr><td>${escape(r.dynamics+' · '+r.family+' · '+r.variant+' · T'+r.timesteps)}</td><td>${escape(r.campaignLabel+' · '+r.stage)}</td><td>${g.runs.map(r=>r.seed).join(', ')} (${g.runs.length})</td><td>${spread(g.mean,g.sd)}</td><td>${tests.length?spread(average(tests),deviation(tests)):'—'}</td></tr>`;
    }));
  }
  function render(){drawLegend();experimentTable();endpoints();hints();specs();seeds();drawCharts();}
  function download(name,content,type){const blob=new Blob([content],{type}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  filterIds.forEach(id=>$(id).addEventListener(id==='search'?'input':'change',()=>{experimentTable();seeds();}));
  $('sort').onchange=experimentTable;
  $('resetFilters').onclick=()=>{filterIds.forEach(id=>$(id).value=id==='status'?'data':'');experimentTable();seeds();};
  $('clearSelection').onclick=()=>{selected=[];message();render();};
  ['accuracyMetric','lossMetric','diagnosticMetric','timeMetric','xmode','epochMin','epochMax'].forEach(id=>$(id).onchange=drawCharts);
  $('resetRange').onclick=()=>{$('epochMin').value=0;$('epochMax').value='';drawCharts();};
  document.querySelectorAll('[data-tab]').forEach(button=>button.onclick=()=>{tab=button.dataset.tab;document.querySelectorAll('[data-tab]').forEach(b=>b.classList.toggle('active',b===button));document.querySelectorAll('[data-view]').forEach(v=>v.hidden=v.dataset.view!==tab);drawCharts();});
  $('exportJson').onclick=()=>download('snn-training-selection.json',JSON.stringify({builtUtc:D.builtUtc,filters:Object.fromEntries(filterIds.map(id=>[id,$(id).value])),runs:runs()},null,2),'application/json');
  $('exportCsv').onclick=()=>{const rows=[['job','campaign','dynamics','encoding','variant','seed','timesteps','stage','status','best_epoch','last_epoch','validation_accuracy','test_accuracy','source'],...filtered().map(r=>[r.job,r.campaign,r.dynamics,r.family,r.variant,r.seed,r.timesteps,r.stage,r.status,r.bestEpoch,r.lastEpoch,r.bestVal,r.test?.accuracy,r.source])];download('snn-experiments.csv','\uFEFF'+rows.map(row=>row.map(v=>'"'+String(v??'').replace(/"/g,'""')+'"').join(';')).join('\r\n'),'text/csv;charset=utf-8');};
  let resize;
  window.addEventListener('resize',()=>{clearTimeout(resize);resize=setTimeout(drawCharts,120);});
  render();
})();
