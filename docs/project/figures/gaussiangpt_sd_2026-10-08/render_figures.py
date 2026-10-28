"""Render two report figures from existing tabular evidence."""
from pathlib import Path
import csv
import html
import subprocess

ROOT = Path(__file__).resolve().parent
NODE = Path('/Users/lief/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
SHARP = '/Users/lief/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp'
BLUE, ORANGE, INK, GREY = '#18689C', '#B95018', '#172D3B', '#647582'

def text(x, y, value, size=19, color=INK, anchor='start', weight=400):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{html.escape(str(value))}</text>'

def line(x1,y1,x2,y2,color='#DDE4E9',width=1,dash=''):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>'

def save(name, height, elements):
    content = f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}"><rect width="1200" height="{height}" fill="white"/><g font-family="Arial, sans-serif">' + ''.join(elements) + '</g></svg>'
    src = ROOT / f'{name}.svg'
    src.write_text(content)
    js = 'const sharp=require(process.argv[1]);sharp(process.argv[2]).png().toFile(process.argv[3]).catch(e=>{console.error(e);process.exit(1)});'
    subprocess.run([str(NODE),'-e',js,SHARP,str(src),str(ROOT/f'{name}.png')],check=True)

def ar_figure():
    rows=list(csv.DictReader((ROOT/'ar_breakdown.csv').open()))
    e=[text(40,48,'Target forward dominates the measured AR window',29,weight=700),text(40,82,'Context approximately 2,048 tokens | values from existing timing export',18,GREY)]
    left,top,scale=285,125,48
    for tick in [0,3,6,9,12,15]:
        x=left+tick*scale
        e.extend([line(x,top,x,325),text(x,352,tick,17,GREY,'middle')])
    colors=[BLUE,ORANGE,GREY]
    for i,row in enumerate(rows):
        y=150+i*76
        val=float(row['median_ms_per_token']); pct=float(row['share_percent_raw'])
        label=row['component'].replace('Outside-phase residual','Other window time')
        e.extend([text(left-18,y+7,label,20,anchor='end'),f'<rect x="{left}" y="{y-15}" width="{val*scale:.3f}" height="32" fill="{colors[i]}" rx="2"/>',text(left+val*scale+12,y+7,f'{val:.3f} ms  |  {pct:.2f}%',19)])
    e.extend([text(646,389,'Time per token (ms)',18,GREY,'middle'),text(40,437,'Total: 13.628 ms/token. Forward includes synchronization within that phase.',18),text(40,468,'Scope: one AR timing window; no whole-pipeline share inferred. Source: ar_breakdown.csv.',17,GREY)])
    save('ar_latency_breakdown',500,e)

def s9_figure():
    rows=list(csv.DictReader((ROOT/'s9_summary.csv').open()))
    e=[text(40,48,'Structured Position drafting leads the S9 trajectory comparison',27,weight=700),text(40,82,'Three seeds per requested length | mean speed ratio with recorded standard deviation',18,GREY)]
    x0,x1,y0,y1=105,1120,130,465
    def ypos(v): return y1-(v-.9)/.6*(y1-y0)
    for tick in [.9,1.,1.1,1.2,1.3,1.4,1.5]:
        yy=ypos(tick)
        e.extend([line(x0,yy,x1,yy, GREY if tick==1 else '#E5E9ED',1.5 if tick==1 else 1,'6 5' if tick==1 else ''),text(x0-14,yy+6,f'{tick:.1f}x',17,GREY,'end')])
    for i,row in enumerate(rows):
        center=235+i*260
        e.append(text(center,498,row['requested_tokens'],19,anchor='middle'))
        for key,dx,col,shape in [('structured',-23,BLUE,'circle'),('d8',23,ORANGE,'square')]:
            mean=float(row[key+'_mean']); std=float(row[key+'_std']); x=center+dx; y=ypos(mean)
            lo,hi=ypos(mean-std),ypos(mean+std)
            e.extend([line(x,hi,x,lo,col,2),line(x-7,hi,x+7,hi,col,2),line(x-7,lo,x+7,lo,col,2)])
            e.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{col}"/>' if shape=='circle' else f'<rect x="{x-6}" y="{y-6}" width="12" height="12" fill="{col}"/>')
            star='*' if key=='d8' and i>=2 else ''
            e.append(text(x,ypos(mean+std)-12,f'{mean:.3f}x{star}',17,col,'middle',700))
    e.extend([text(612,534,'Requested output tokens',18,GREY,'middle'),f'<circle cx="55" cy="572" r="6" fill="{BLUE}"/>',text(72,578,'Structured Position',19,BLUE),f'<rect x="344" y="566" width="12" height="12" fill="{ORANGE}"/>',text(366,578,'D8 Position-only',19,ORANGE),text(655,578,'1x = ordinary Target decoding',18,GREY),text(40,618,'* D8 includes seed 2 stopping at EOS after 1,753 tokens for both longer requests.',18),text(40,650,'Seed trajectories, not independent scenes. Historical timing conditions require further review.',17,GREY),text(40,680,'Source: S9 historical summary, transcribed in s9_summary.csv. Error bars are not confidence intervals.',16,GREY)])
    save('s9_speedup_by_length',710,e)

if __name__=='__main__':
    ar_figure()
    s9_figure()
