const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');

let memmeNode = null;
let localStore = null;

// Try to load memme-node for local mode
try {
  memmeNode = require('../../crates/memme-node');
  console.log('memme-node loaded — local mode available');
} catch (e) {
  console.log('memme-node not available — remote mode only');
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    backgroundColor: '#0a0a0f',
    titleBarStyle: 'hiddenInset',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  win.loadFile('renderer/index.html');
  win.setTitle('MemMe Desktop');
}

// IPC: Check if local mode is available
ipcMain.handle('memme:hasLocal', () => !!memmeNode);

// IPC: Initialize local store
ipcMain.handle('memme:initLocal', async (_, dbPath) => {
  if (!memmeNode) throw new Error('memme-node not available');
  try {
    localStore = memmeNode.MemoryStore.new_mock(dbPath || ':memory:', 384);
    return true;
  } catch (e) {
    return { error: e.message };
  }
});

// IPC: Local add memory
ipcMain.handle('memme:localAdd', async (_, content, userId, agentId, runId, metadata) => {
  if (!localStore) throw new Error('Local store not initialized');
  return await localStore.add(content, userId, agentId, runId, metadata);
});

// IPC: Local search
ipcMain.handle('memme:localSearch', async (_, query, userId, agentId, runId, limit, threshold) => {
  if (!localStore) throw new Error('Local store not initialized');
  return await localStore.search(query, userId, agentId, runId, limit, threshold);
});

// IPC: Local list
ipcMain.handle('memme:localList', async (_, userId, agentId, runId, limit) => {
  if (!localStore) throw new Error('Local store not initialized');
  return await localStore.list(userId, agentId, runId, limit);
});

// IPC: Local delete
ipcMain.handle('memme:localDelete', async (_, id) => {
  if (!localStore) throw new Error('Local store not initialized');
  return await localStore.delete(id);
});

// IPC: Local hybrid search
ipcMain.handle('memme:localHybridSearch', async (_, query, userId, agentId, runId, limit) => {
  if (!localStore) throw new Error('Local store not initialized');
  return await localStore.hybrid_search(query, userId, agentId, runId, limit);
});

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
