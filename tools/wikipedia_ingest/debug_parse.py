#!/usr/bin/env python3
"""Debug Wikipedia parsing to see what's being filtered out"""

import bz2
import xml.etree.ElementTree as ET
import re
import mwparserfromhell

dump_path = "data/simplewiki-latest-pages-articles.xml.bz2"
namespace = "{http://www.mediawiki.org/xml/export-0.11/}"

def wikitext_to_plain(wikitext):
    """Convert wikitext to plain text"""
    try:
        parsed = mwparserfromhell.parse(wikitext)
        text = parsed.strip_code()
        text = re.sub(r'\n\n+', '\n\n', text)
        return text.strip()
    except Exception as e:
        print(f"Parse error: {e}")
        # Fallback
        text = re.sub(r'\[\[([^\]|]+\|)?([^\]]+)\]\]', r'\2', wikitext)
        text = re.sub(r'\{\{[^}]+\}\}', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        return text.strip()

count = 0
pages_seen = 0
with bz2.open(dump_path, 'rb') as f:
    for event, elem in ET.iterparse(f, events=('end',)):
        if elem.tag == f'{namespace}page':
            pages_seen += 1
            print(f"\n--- Page {pages_seen} ---")
            print(f"Tag: {elem.tag}")

            title_elem = elem.find(f'{namespace}title')
            print(f"Title elem: {title_elem}")
            if title_elem is not None:
                print(f"Title text: {title_elem.text}")

            revision = elem.find(f'{namespace}revision')
            print(f"Revision elem: {revision}")

            if title_elem is not None and revision is not None:
                title = title_elem.text
                text_elem = revision.find(f'{namespace}text')
                print(f"Text elem: {text_elem}")

                if text_elem is not None:
                    print(f"Text elem attributes: {text_elem.attrib}")
                    print(f"Text elem text: {text_elem.text is not None}")
                    if text_elem.text:
                        print(f"Text length: {len(text_elem.text)}")

                if text_elem is not None and text_elem.text:
                    wikitext = text_elem.text
                    plain_text = wikitext_to_plain(wikitext)

                    print(f"\n{'='*60}")
                    print(f"Article: {title}")
                    print(f"Wikitext length: {len(wikitext)}")
                    print(f"Plain text length: {len(plain_text)}")
                    print(f"First 200 chars of plain text:")
                    print(plain_text[:200])
                    print(f"{'='*60}")

                    count += 1
                    if count >= 5:
                        break

            elem.clear()

            if pages_seen >= 10:
                print("\nStopping after 10 pages for debugging")
                break

print(f"\nPages seen: {pages_seen}")
print(f"Articles processed: {count}")
