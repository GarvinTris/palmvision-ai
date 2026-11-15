// Isi dengan nilai dari Supabase Dashboard → Project Settings → API.
// ANON KEY memang publik (aman di frontend) karena akses data dijaga oleh Row Level Security di supabase/schema.sql.
// JANGAN pernah menaruh "service_role" key di sini.
window.PALM_CONFIG = {
  SUPABASE_URL: "https://uykvnhmkebehnmocftjx.supabase.co",
  SUPABASE_ANON_KEY: "sb_publishable_U-p28TfmBk_Lg-OI3eaL8A_31bp15XB",
  STORAGE_BUCKET: "inspections",
};
  