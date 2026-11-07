// Shared by portal.js and app.js: reading numbers typed or pasted into the page.
'use strict';

// A number as typed or pasted: accepts the typographic minus (\u2212) and dashes, a decimal comma, spaces
// around it, a degree sign and a trailing N/S/E/W (S and W make it negative). NaN if it isn't one number (a
// space inside the number itself is taken as a typo, not ignored).
function parseNum(text) {
  let s = String(text ?? '').replace(/[\u2212\u2010-\u2015\uFE63\uFF0D]/g, '-').replace(/\u00a0/g, ' ').trim()
    .replace(/^([+-])\s+/, '$1').replace(/\s*\u00b0\s*/g, '');
  let sign = 1;
  const hemi = /^(.*?)\s*([NSEWnsew])$/.exec(s);
  if (hemi) { s = hemi[1]; if (/[SWsw]/.test(hemi[2])) sign = -1; }
  if (/^[+-]?\d+,\d+$/.test(s)) s = s.replace(',', '.');
  return /^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/i.test(s) ? sign * parseFloat(s) : NaN;
}

// "39.459816, -119.958273" (or with a space or semicolon between) -> [lat, lon]; null if it isn't a valid pair.
function parsePair(text) {
  const parts = String(text ?? '').trim().split(/\s*[,;]\s*|\s+(?=[-+\u2212\d.])/);
  if (parts.length !== 2) return null;
  const v = parts.map(parseNum);
  return v.every(isFinite) && Math.abs(v[0]) <= 90 && Math.abs(v[1]) <= 180 ? v : null;   // a real lat/lon only
}

// Read the named inputs as numbers; throws an error naming the first field that isn't one.
function readNums(fields) {
  return fields.map(([id, label]) => {
    const raw = document.getElementById(id).value, v = parseNum(raw);
    if (!isFinite(v)) throw new Error(`${label}: "${raw}" is not a number.`);
    return v;
  });
}

// A "lat, lon" pair pasted into a latitude box fills that box and the longitude box.
function splitPairInto(latId, lonId) {
  const lat = document.getElementById(latId);
  const fill = () => {
    const p = parsePair(lat.value);
    if (p) { lat.value = p[0]; document.getElementById(lonId).value = p[1]; }
  };
  lat.addEventListener('change', fill);
  lat.addEventListener('paste', () => setTimeout(fill, 0));
}
