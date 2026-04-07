const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('memme', {
  // Check if local mode (memme-node) is available
  hasLocal: () => ipcRenderer.invoke('memme:hasLocal'),

  // Initialize local store
  initLocal: (dbPath) => ipcRenderer.invoke('memme:initLocal', dbPath),

  // Local mode operations
  localAdd: (content, userId, agentId, runId, metadata) =>
    ipcRenderer.invoke('memme:localAdd', content, userId, agentId, runId, metadata),

  localSearch: (query, userId, agentId, runId, limit, threshold) =>
    ipcRenderer.invoke('memme:localSearch', query, userId, agentId, runId, limit, threshold),

  localList: (userId, agentId, runId, limit) =>
    ipcRenderer.invoke('memme:localList', userId, agentId, runId, limit),

  localDelete: (id) =>
    ipcRenderer.invoke('memme:localDelete', id),

  localHybridSearch: (query, userId, agentId, runId, limit) =>
    ipcRenderer.invoke('memme:localHybridSearch', query, userId, agentId, runId, limit),
});
