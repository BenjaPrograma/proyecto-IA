/* 1027V8 harmless F01 execution marker. No network, storage, credentials, or session data. */
(function () {
  "use strict";
  var marker = "V8F01JSD1027xQ7";
  var script = document.currentScript;
  var prior = window.__f01V8ThirdParty1027;
  var executions = prior && prior.executions ? prior.executions : [];
  var record = {
    marker: marker,
    origin: location.origin,
    realm: window === window.top ? "top" : "nested",
    script_url: script ? script.src : null,
    script_crossOrigin: script ? script.crossOrigin : null,
    execution_index: executions.length + 1,
    time_ms: Date.now()
  };
  executions.push(record);
  window.__f01V8ThirdParty1027 = {
    marker: marker,
    count: executions.length,
    origin: record.origin,
    realm: record.realm,
    executions: executions
  };
  function markBody() {
    if (!document.body) return;
    document.body.dataset.f01V8ThirdParty1027 = marker;
    document.body.dataset.f01V8ThirdParty1027Origin = record.origin;
    document.body.dataset.f01V8ThirdParty1027Realm = record.realm;
    document.body.dataset.f01V8ThirdParty1027Count = String(executions.length);
    document.body.dataset.f01V8ThirdParty1027Crossorigin = String(record.script_crossOrigin);
  }
  markBody();
  if (!document.body) document.addEventListener("DOMContentLoaded", markBody, { once: true });
  if (typeof window.__v8rec === "function") {
    window.__v8rec(JSON.stringify({ k: "third_party_marker", data: record }));
  }
})();
