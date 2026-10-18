import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend, Rate } from 'k6/metrics';
import { textSummary } from 'https://jslib.k6.io/k6-summary/0.1.0/index.js';

const searchMs = new Trend('search_ms', true);
const under3s = new Rate('search_under_3s');
const BASE = __ENV.BASE_URL || 'http://host.docker.internal:8000';
const KEY = __ENV.API_KEY;
const QUERIES = JSON.parse(open('./queries.json'));
const VUS = parseInt(__ENV.VUS || '5');
const DURATION = __ENV.DURATION || '60s';

export const options = {
  summaryTrendStats: ['avg', 'min', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
  scenarios: {
    search: { executor: 'constant-vus', vus: VUS, duration: DURATION },
  },
  thresholds: {
    http_req_failed: ['rate<0.01'],
    search_ms: ['p(95)<2800', 'p(99)<3000'],
    search_under_3s: ['rate>0.98'],
  },
};

const H = { 'Content-Type': 'application/json', 'X-API-Key': KEY };

export default function () {
  const q = QUERIES[(__VU * 7919 + __ITER) % QUERIES.length];
  const r = http.post(`${BASE}/search`, JSON.stringify({ query: q, top_k: 10 }), { headers: H, tags: { endpoint: 'search' } });
  searchMs.add(r.timings.duration);
  under3s.add(r.timings.duration < 3000);
  check(r, {
    'search 200': (x) => x.status === 200,
    'search has results': (x) => x.status === 200 && JSON.parse(x.body).results.length >= 1,
  });
  if (__ITER % 10 === 0) {
    const e = http.post(`${BASE}/embed`, JSON.stringify({ texts: [q] }), { headers: H, tags: { endpoint: 'embed' } });
    check(e, { 'embed 200': (x) => x.status === 200 });
    const rr = http.post(`${BASE}/rerank`, JSON.stringify({ query: q, documents: ['Áo bà ba', 'Chợ nổi', 'Lễ hội'], top_n: 2 }), { headers: H, tags: { endpoint: 'rerank' } });
    check(rr, { 'rerank 200': (x) => x.status === 200 });
  }
  sleep(0.2);
}

export function handleSummary(data) {
  return {
    '/reports/loadtest-summary.json': JSON.stringify(data, null, 2),
    stdout: textSummary(data, { indent: ' ', enableColors: false }),
  };
}
