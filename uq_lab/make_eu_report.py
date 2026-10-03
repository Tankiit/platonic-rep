"""Describe EU-specific runs without treating shared-encoder pairs as independent."""
import argparse
import json
from pathlib import Path
import pandas as pd


def report(runs, output):
    lines=['# EU-specific experiment results', '',
           'Exploratory Gaussian-regression EU on frozen features. These runs do not establish classification EU or acquisition utility.', '',
           'The tables use constant AU, prior variance 1, and the largest training fraction. Entries are mean correlations across seeds with the seed range. Pair observations share encoders; ranges are sensitivity summaries, not confidence intervals.', '']
    image_frames=[]
    for root in runs:
        manifest=json.loads((root/'manifest.json').read_text())
        predictions=pd.read_csv(root/'metric_prediction.csv')
        pairs=pd.read_csv(root/'pairs.csv')
        diagnostics=pd.read_csv(root/'diagnostics.csv')
        if manifest['mode']=='features':
            frame=predictions.copy()
            frame['head']=manifest['head']
            frame['heldout_class']=manifest['heldout']
            image_frames.append(frame)
        selected=predictions[(predictions.noise_condition=='homoscedastic') &
                             (predictions.prior==1) & (predictions.fraction==1)]
        lines.extend([f'## {root.name}', '',
                      f"Head: {manifest['head']}; mode: {manifest['mode']}; seeds: {manifest['seeds']}; held-out class: {manifest['heldout'] if manifest['mode']=='features' else 'controlled coverage shift'}.", '',
                      f'{len(pairs)} pair/configuration records; {len(diagnostics)} variance checks; all contraction checks passed: {bool(diagnostics.monotonic_pass.all())}.', '',
                      '| Metric | EU ranking | Acquisition overlap | ID-only EU | Held-out-only EU |',
                      '|---|---|---|---|---|'])
        for metric, group in selected.groupby('metric'):
            values=[]
            for target in ('eu_agreement','acquisition_overlap','eu_id','eu_heldout'):
                v=group[group.target==target].correlation.dropna()
                values.append(f'{v.mean():.2f} [{v.min():.2f}, {v.max():.2f}]' if len(v) else 'undefined')
            lines.append('| '+metric+' | '+' | '.join(values)+' |')
        sat=selected.groupby('metric').metric_span.max()
        warnings=sat[sat<1e-4]
        if len(warnings):
            lines.extend(['', 'Near-saturated metrics (span < 1e-4): '+', '.join(warnings.index)+'. Their rank correlations can depend on tiny numerical differences and should not support a metric-superiority claim.'])
        d=diagnostics[(diagnostics.noise_condition=='homoscedastic') &
                      (diagnostics.prior==1)&(diagnostics.fraction==1)]
        ratio=d.mean_eu_heldout/d.mean_eu_id
        lines.extend(['', f'Held-out/ID mean EU ratio across views/seeds: {ratio.min():.2f}–{ratio.max():.2f}.', '',
                      f"Source SHA-256: `{manifest['source_sha256']}`.", ''])
    if image_frames:
        frame=pd.concat(image_frames,ignore_index=True)
        selected=frame[(frame.noise_condition=='homoscedastic')&(frame.prior==1)&(frame.fraction==1)]
        aggregated=['## Across withheld classes', '',
            'Each cell averages seeds within a withheld class, then reports the mean and min–max across classes. These are descriptive sensitivity summaries, not confidence intervals.', '']
        for head,group in selected.groupby('head'):
            aggregated.extend([f'### {head}', '',
                '| Metric | EU ranking | Acquisition overlap | ID-only EU | Held-out-only EU |',
                '|---|---|---|---|---|'])
            for metric,mgroup in group.groupby('metric'):
                vals=[]
                for target in ('eu_agreement','acquisition_overlap','eu_id','eu_heldout'):
                    v=mgroup[mgroup.target==target].groupby('heldout_class').correlation.mean().dropna()
                    vals.append(f'{v.mean():.2f} [{v.min():.2f}, {v.max():.2f}]' if len(v) else 'undefined')
                aggregated.append('| '+metric+' | '+' | '.join(vals)+' |')
            aggregated.append('')
        lines=lines[:6]+aggregated+lines[6:]
    lines.extend(['## What remains', '',
                  'The image results are a small pilot. Extend to the full image caches, inspect leave-one-encoder-out sensitivity, validate acquisition utility on learning curves, and add a classification EU estimator with an independently supported likelihood. Known AU is prescribed, not estimated from human disagreement.'])
    output.write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,default=Path('EU_RESULTS.md'))
    args=parser.parse_args()
    report(args.runs,args.output)
