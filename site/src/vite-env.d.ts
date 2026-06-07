/// <reference types="vite/client" />

// The fork's `lib/index-basic.js` entry isn't in DefinitelyTyped's
// `plotly.js` types. We import it dynamically + cast to `typeof import('plotly.js')`
// at the call site (see `src/App.tsx`), so an opaque module shape is fine here.
declare module 'plotly.js/lib/index-basic.js'
