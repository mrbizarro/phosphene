// Phosphene's push service worker. One job: turn a push message into a
// notification when the tab is closed, and focus or open the panel when it is
// clicked. It caches nothing and intercepts no request — the panel is local.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));
self.addEventListener('push', (e) => {
  let d = {};
  try { d = e.data ? e.data.json() : {}; } catch (err) { d = { title: 'Phosphene', body: (e.data && e.data.text()) || '' }; }
  e.waitUntil(self.registration.showNotification(d.title || 'Phosphene', {
    body: d.body || '', tag: d.tag || 'phos', renotify: true,
    icon: '/assets/phosphene_favicon_256.png',
  }));
});
self.addEventListener('notificationclick', (e) => {
  e.notification.close();
  e.waitUntil(self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then((list) => {
    for (const c of list) { if ('focus' in c) return c.focus(); }
    if (self.clients.openWindow) return self.clients.openWindow('/');
    return null;
  }));
});
