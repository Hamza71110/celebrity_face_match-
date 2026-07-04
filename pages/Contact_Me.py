import streamlit as st

st.set_page_config(
    page_title="Contact Me · Celebrity Face Matcher",
    page_icon="📬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

LINKEDIN = "https://www.linkedin.com/in/hamza-akbar-75799b375/"
GITHUB = "https://github.com/Hamza71110"
EMAIL = "muhammadhamzaakbar10@gmail.com"

st.markdown("""
<style>
.stApp{ background: linear-gradient(135deg,#0F172A,#111827,#1E293B); color:white; }
.block-container{ max-width:820px; padding-top:2.5rem; }
.stApp, .stApp p, .stApp span, .stApp div, .stApp a, .stApp h1{ font-weight:700 !important; }
#MainMenu, footer, header{ visibility:hidden; }
[data-testid="stSidebar"], [data-testid="collapsedControl"],
[data-testid="stSidebarNav"]{ display:none !important; }

.contact-hero{ text-align:center; margin-bottom:8px; }
.contact-hero h1{ font-size:clamp(30px,7vw,52px); font-weight:800; margin:0; }
.contact-hero p{ color:#9CA3AF; font-size:clamp(14px,2.6vw,20px); margin-top:6px; }

.wave{ display:inline-block; animation:wave 2s ease-in-out infinite; transform-origin:70% 70%; }
@keyframes wave{ 0%,60%,100%{transform:rotate(0);} 10%{transform:rotate(14deg);}
  20%{transform:rotate(-8deg);} 30%{transform:rotate(14deg);} 40%{transform:rotate(-4deg);}
  50%{transform:rotate(10deg);} }

.card-grid{ display:flex; flex-direction:column; gap:16px; margin-top:26px; }
.contact-card{ display:flex; align-items:center; gap:18px; text-decoration:none;
  background:#1E293B; border:1px solid #334155; border-left:6px solid #8B5CF6;
  border-radius:18px; padding:18px 22px; transition:.25s;
  box-shadow:0 10px 25px rgba(0,0,0,.35); }
.contact-card:hover{ transform:translateY(-3px); border-left-color:#A855F7;
  box-shadow:0 16px 34px rgba(168,85,247,.35); }
.contact-card .ico{ font-size:34px; }
.contact-card .txt{ text-align:left; }
.contact-card .txt .t{ color:#fff; font-weight:700; font-size:19px; }
.contact-card .txt .s{ color:#9CA3AF; font-size:14px; word-break:break-all; }

.back-link{ display:inline-block; margin-top:30px; color:#A855F7; text-decoration:none;
  font-weight:700; font-size:16px; }
.back-link:hover{ color:#C084FC; }
.app-footer{ text-align:center; color:#9CA3AF; font-size:16px; margin-top:36px; }
.app-footer b{ color:#E5E7EB; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="contact-hero">
  <h1><span class="wave">👋</span> Get in touch</h1>
  <p>Have feedback or want to collaborate? Reach me here.</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="card-grid">

  <a class="contact-card" href="{LINKEDIN}" target="_blank">
    <span class="ico">💼</span>
    <span class="txt"><div class="t">LinkedIn</div>
      <div class="s">Hamza Akbar</div></span>
  </a>

  <a class="contact-card" href="{GITHUB}" target="_blank">
    <span class="ico">🐙</span>
    <span class="txt"><div class="t">GitHub</div>
      <div class="s">github.com/Hamza71110</div></span>
  </a>

  <a class="contact-card" href="mailto:{EMAIL}">
    <span class="ico">✉️</span>
    <span class="txt"><div class="t">Email</div>
      <div class="s">{EMAIL}</div></span>
  </a>

</div>

<a class="back-link" href="/" target="_self">← Back to Face Matcher</a>

<div class="app-footer">Made with ❤️ by <b>Hamzurna</b></div>
""", unsafe_allow_html=True)
