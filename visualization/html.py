"""Small standalone reports; every image/video byte is embedded, no navigation index."""
import base64
from html import escape
from pathlib import Path


def data_uri(path):
    path = Path(path)
    mime = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.mp4': 'video/mp4'}[path.suffix.lower()]
    return f'data:{mime};base64,'+base64.b64encode(path.read_bytes()).decode('ascii')


def media_report(path, title, note, media):
    """media is an ordered sequence of (caption, image-or-video-path)."""
    path = Path(path)
    if path.stem.lower() == 'index':
        raise ValueError('Deliver a standalone report, not an index HTML')
    body = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
            f'<title>{escape(title)}</title>',
            '<style>body{background:#10151d;color:#e4eaf4;font:17px system-ui;margin:24px}img,video{max-width:100%;max-height:90vh}p{line-height:1.6}section{margin:24px 0}</style>',
            f'<h1>{escape(title)}</h1><p>{escape(note)}</p>']
    for caption, source in media:
        source = Path(source)
        tag = 'video controls preload="metadata"' if source.suffix.lower() == '.mp4' else 'img'
        end = '</video>' if source.suffix.lower() == '.mp4' else ''
        body.append(f'<section><h2>{escape(caption)}</h2><{tag} src="{data_uri(source)}">{end}</section>')
    body.append('</html>')
    path.write_text('\n'.join(body))

