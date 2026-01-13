const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:5000';

console.log('API_BASE configured as:', API_BASE);

export async function getJSON<T>(path: string): Promise<T> {
  const url = `${API_BASE}${path}`;
  console.log(`[API] Fetching: ${url}`);
  try {
    const res = await fetch(url);
    if (!res.ok) {
      console.error(`[API] Error ${res.status} from ${url}`);
      throw new Error(`HTTP ${res.status}`);
    }
    const data = await res.json() as T;
    console.log(`[API] Success from ${path}:`, data);
    return data;
  } catch (error) {
    console.error(`[API] Failed to fetch ${url}:`, error);
    throw error;
  }
}
