# ระบบวิเคราะห์ไฟล์ FITS แบบ Multi-Agent

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.118-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Groq](https://img.shields.io/badge/LLM-Groq-F55036)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](README.md) · **ภาษาไทย**

> ระบบ backend แบบ multi-agent สำหรับวิเคราะห์และตีความข้อมูลอนุกรมเวลาทางดาราศาสตร์รังสีเอกซ์ (ไฟล์ FITS) โดยอัตโนมัติ

พัฒนาสำหรับ **สาขาวิชาฟิสิกส์ มหาวิทยาลัยเทคโนโลยีสุรนารี (มทส.)** ร่วมกับ **สถาบันวิจัยดาราศาสตร์แห่งชาติ (องค์การมหาชน) (NARIT)** เพื่อรองรับงานวิเคราะห์เชิงเวลา (timing analysis) ของข้อมูลจากกล้องโทรทรรศน์ XMM-Newton (เช่น AGN *IRAS 13224-3809*)

---

## ภาพรวม

ผู้ใช้ (นักวิจัย) อัปโหลดไฟล์ FITS ที่เป็น light curve แล้วพิมพ์คำถามเป็นภาษาธรรมชาติ ระบบจะจำแนกประเภทคำขอ เลือกรันการวิเคราะห์เชิงตัวเลขที่เหมาะสม (สถิติ, PSD, การ fit โมเดล) ตีความผลลัพธ์ในเชิงฟิสิกส์ดาราศาสตร์ด้วย LLM แล้วส่งคำตอบที่เรียบเรียงแล้วพร้อมกราฟกลับไป — พร้อมทั้งบันทึกประวัติการสนทนาและการวิเคราะห์ทั้งหมด

ระบบออกแบบเป็น **เอเจนต์เฉพาะทางหลายตัวที่ประสานงานผ่าน orchestrator แบบ dynamic routing** (รูปแบบ supervisor/router) แทนที่จะเป็นโมเดลเดี่ยวก้อนใหญ่

```
คำถามผู้ใช้ + ไฟล์ FITS
        │
        ▼
┌───────────────────────┐
│   Orchestrator        │  async queue · workers · จำกัด concurrency
│  (dynamic routing)    │
└───────────┬───────────┘
            ▼
   ┌────────────────────┐
   │ Classification &   │  จำแนกเจตนา + สกัดพารามิเตอร์ (LLM)
   │ Parameter Agent    │
   └─────────┬──────────┘
             │  เลือกเส้นทาง (routing strategy)
   ┌─────────┼───────────────────────┐
   ▼         ▼                        ▼
analysis   interpretation          mixed
   │         │                    (วิเคราะห์ → ตีความ)
   ▼         ▼                        ▼
┌────────────────┐   ┌────────────────────┐
│ Analysis Agent │   │ Interpretation     │  ให้เหตุผลเชิงฟิสิกส์ (LLM)
│ (เชิงตัวเลข)    │   │ Agent              │
└────────┬───────┘   └─────────┬──────────┘
         └──────────┬──────────┘
                    ▼
            ┌───────────────┐
            │ Rewrite Agent │  เรียบเรียงคำตอบสุดท้าย (LLM)
            └───────┬───────┘
                    ▼
        คำตอบ + กราฟ + ประวัติ (PostgreSQL)
```

---

## จุดเด่นของระบบ

- **Dynamic intent-based routing** — ตัวจำแนกจะเลือก 1 ใน 3 เส้นทาง:
  - `analysis` — ประมวลผลเชิงตัวเลขล้วน
  - `interpretation` — ตอบคำถามเชิงแนวคิด/ฟิสิกส์ดาราศาสตร์
  - `mixed` — วิเคราะห์ก่อน แล้วตีความเชิงฟิสิกส์ต่อ
- **การวิเคราะห์อนุกรมเวลา FITS** — สถิติเชิงพรรณนา, Power Spectral Density (PSD), และการ fit โมเดล **power-law / bending-power-law** (ด้วย SciPy) พร้อมสร้างกราฟอัตโนมัติ
- **Parameter inheritance** — การถามต่อเนื่อง (เช่น "*ทำอีกครั้งด้วย A0=5*") จะนำพารามิเตอร์เดิมมาใช้ซ้ำและแก้เฉพาะค่าที่ระบุ โดยแยกขอบเขตตามไฟล์หรือตาม session
- **Async orchestration** — คิวคำขอพร้อม background workers และจำกัดจำนวนงานพร้อมกัน (semaphore) ตามระดับของ LLM
- **API สองเวอร์ชัน (v1 & v2)** — v2 เพิ่ม pagination, การบีบอัด GZIP และ **Server-Sent Events (SSE)** สำหรับสตรีมสถานะแบบเรียลไทม์
- **บันทึกข้อมูลครบถ้วน** — โครงสร้าง PostgreSQL (6 ตาราง) ครอบคลุมผู้ใช้ ไฟล์ session ประวัติการวิเคราะห์ และกราฟที่สร้างขึ้น
- **Prompt engineering เพื่อความน่าเชื่อถือ** — ใช้ prompt ที่บังคับ output แบบมีโครงสร้าง และปรับคำตอบตามระดับความเชี่ยวชาญ (beginner → expert) เพื่อลด hallucination

---

## เทคโนโลยีที่ใช้

| ส่วน | เทคโนโลยี |
|------|-----------|
| API / Web | FastAPI, Uvicorn |
| ฐานข้อมูล | PostgreSQL (async ผ่าน `asyncpg` + SQLAlchemy 2.0) |
| การประสาน LLM | LangChain, Groq (`langchain-groq`) |
| เชิงตัวเลข | NumPy, SciPy, เครื่องมือ FITS/Astropy, Matplotlib |
| จัดการแพ็กเกจ | `uv` (PEP 621 `pyproject.toml`) |

### โมเดล LLM

เอเจนต์ทำงานบน **โมเดลที่โฮสต์บน Groq** ปรับแยกตามเอเจนต์ได้ใน [`app/core/config.py`](app/core/config.py):

| เอเจนต์ | โมเดลเริ่มต้น |
|---------|--------------|
| Classification & Parameter | `llama-3.3-70b-versatile` |
| Interpretation | `openai/gpt-oss-120b` |
| Rewrite | `llama-3.3-70b-versatile` |

> **ที่มา:** เดิมระบบเป็นสถาปัตยกรรมแบบ hybrid ที่ใช้โมเดล **AstroSage-Llama-3.1-8B** (โฮสต์ในเครื่อง) ร่วมกับ **OpenAI GPT** สำหรับการตีความ ปัจจุบันส่วนตีความ (`app/services/astrosage/`) ยังคงอยู่แต่ย้ายมาใช้โมเดลผ่าน Groq แล้ว — คีย์ OpenAI เป็นของเดิมและไม่จำเป็นต้องใช้อีกต่อไป

---

## โครงสร้างโปรเจกต์

```
app/
├── main.py                     # FastAPI app, lifespan, ลงทะเบียน router
├── orchestration/
│   └── orchestrator.py         # DynamicWorkflowOrchestrator (คิว, routing, บันทึกข้อมูล)
├── agents/
│   ├── classification_parameter/   # เอเจนต์จำแนกเจตนา + สกัดพารามิเตอร์
│   ├── analysis/                   # เอเจนต์วิเคราะห์เชิงตัวเลข + capabilities
│   └── rewrite/                    # เรียบเรียงคำตอบสุดท้าย + ตั้งชื่อ session
├── services/
│   ├── astrosage/                  # LLM interpretation client & prompt building
│   └── conversation_service.py     # session, ข้อความ, บันทึก workflow
├── tools/                      # statistics, psd, fitting, plotting, ตัวโหลด FITS
├── api/
│   ├── v1/                      # API เดิม (response เต็ม)
│   └── v2/                      # API ที่ปรับให้เบา (pagination, SSE, GZIP)
├── db/                         # SQLAlchemy models + async engine
└── core/                       # config, constants
scripts/
└── init_db.sql                 # ติดตั้ง extension + schema เริ่มต้นของ PostgreSQL
```

---

## เริ่มต้นใช้งาน

### สิ่งที่ต้องมี

- Python **3.12+**
- [`uv`](https://docs.astral.sh/uv/) (หรือ pip)
- Docker และ Docker Compose (สำหรับ PostgreSQL)
- **Groq API key** — https://console.groq.com

### 1. Clone และติดตั้ง dependencies

```bash
git clone <repository-url>
cd multi-agent-fits-develop-02-project

uv sync            # หรือ: pip install -r requirements.txt
```

### 2. ตั้งค่า environment

สร้างไฟล์ `.env` ที่ root ของโปรเจกต์:

```dotenv
# LLM
GROQ_API_KEY=your_groq_api_key_here

# ฐานข้อมูล — โปรเจกต์นี้ใช้พอร์ต 5433 บน host (พอร์ต 5432 ถูกอีกโปรเจกต์ใช้อยู่)
POSTGRES_PORT=5433
DATABASE_URL=postgresql+asyncpg://fits_user:fits_password@localhost:5433/fits_analysis_db

# Auth
SECRET_KEY=เปลี่ยนเป็น_secret_แบบสุ่ม

# Storage (ไม่บังคับ — ค่าเริ่มต้นตามนี้)
FITSFILES_DIR=storage/fitsfiles
PLOTS_DIR=storage/plots
```

### 3. รันฐานข้อมูล

```bash
docker compose up -d postgres
```

ไฟล์ `scripts/init_db.sql` จะรันอัตโนมัติในครั้งแรกเพื่อสร้าง extension ที่จำเป็น (`uuid-ossp`, `pg_trgm`) และ schema — มี pgAdmin (ทางเลือก) ที่ `http://localhost:5050`

### 4. รัน API

```bash
uv run uvicorn app.main:app --reload --port 8003
```

ตารางของแอปพลิเคชันจะถูกสร้างอัตโนมัติตอนเริ่มระบบ

---

## API

| บริการ | URL |
|--------|-----|
| Swagger UI | http://localhost:8003/docs |
| ReDoc | http://localhost:8003/redoc |
| Health check | http://localhost:8003/health |
| API v1 | `http://localhost:8003/api/v1` |
| API v2 (แนะนำ) | `http://localhost:8003/api/v2` |

### ขั้นตอนการใช้งานทั่วไป

1. `POST /api/v2/auth/...` — เข้าสู่ระบบ
2. `POST /api/v2/files` — อัปโหลดไฟล์ FITS
3. `POST /api/v2/analyze` — ส่งคำถาม → ได้ `task_id` กลับมา
4. `GET /api/v2/analyze/{task_id}/stream` — สตรีมความคืบหน้าด้วย SSE **หรือ** poll ที่ `GET /api/v2/analyze/{task_id}/status`
5. `GET /api/v2/analyze/{task_id}/result` — ดึงคำตอบสุดท้าย + กราฟ

### ตัวอย่าง: ส่งคำขอวิเคราะห์

**Request** — `POST /api/v2/analyze` (ต้องมี Bearer token)

```json
{
  "query": "Fit the PSD with a bending power law using 5000 bins and a break frequency of 0.02, then explain how the spectral slopes relate to accretion disk turbulence.",
  "fits_file_id": "eb536ea0-cce7-479f-9153-7b3bc189d71d",
  "session_id": null,
  "user_expertise": "intermediate"
}
```

**Response** — `202` (เข้าคิวแล้ว)

```json
{
  "task_id": "9f1c2b7a-1d3e-4a2f-b8c0-5e6d7f8a9b01",
  "session_id": "3a7e5c9d-2b4f-4e1a-9c8d-0f1e2d3c4b5a",
  "status": "queued"
}
```

### ตัวอย่าง: poll สถานะ (แบบเบา)

**Request** — `GET /api/v2/analyze/{task_id}/status`

```json
{
  "task_id": "9f1c2b7a-1d3e-4a2f-b8c0-5e6d7f8a9b01",
  "status": "in_progress",
  "progress": "50%",
  "current_step": "analysis",
  "error": null
}
```

### ตัวอย่าง: อัปเดตแบบเรียลไทม์ (SSE)

**Request** — `GET /api/v2/analyze/{task_id}/stream`

```
data: {"task_id": "9f1c2b7a...", "status": "in_progress", "progress": "30%", "current_step": "classification"}

data: {"task_id": "9f1c2b7a...", "status": "in_progress", "progress": "70%", "current_step": "astrosage"}

data: {"task_id": "9f1c2b7a...", "status": "completed", "progress": "100%", "current_step": "rewrite"}
```

### ตัวอย่าง: ดึงผลลัพธ์

**Request** — `GET /api/v2/analyze/{task_id}/result`

```json
{
  "task_id": "9f1c2b7a-1d3e-4a2f-b8c0-5e6d7f8a9b01",
  "status": "completed",
  "content": "การ fit ด้วย bending power-law ให้ความชันช่วงความถี่ต่ำ ~1.0 และชันขึ้นเป็น ~2.3 เหนือจุดหักที่ f_b ≈ 0.02 Hz ซึ่งสอดคล้องกับ ...",
  "plots": [
    {
      "plot_id": "b2c3d4e5-...",
      "plot_type": "bending_power_law",
      "plot_url": "/api/v1/plots/bending_power_law/9f1c2b7a_bpl.png",
      "title": "Bending Power Law Fit",
      "created_at": "2026-07-11T09:15:42.123456"
    }
  ],
  "completed_at": "2026-07-11T09:15:43.987654"
}
```

---

## ประเภทการวิเคราะห์

| ประเภท | คำอธิบาย |
|--------|----------|
| `statistics` | สถิติเชิงพรรณนา (mean, median, std, min, max, percentile) |
| `psd` | Power Spectral Density ของ light curve |
| `power_law` | fit power-law อย่างง่าย: `PSD(f) = A/f^b + n` |
| `bending_power_law` | fit bending power-law: `PSD(f) = A / [f (1 + (f/f_b)^(sh-1))] + n` |
| `metadata` | สกัด metadata จาก FITS header / HDU (รวมให้เสมอ) |

---

## การทดสอบ

```bash
uv run pytest
```

เทสต์ระดับเอเจนต์อยู่ข้างๆ ตัวเอเจนต์ (เช่น `app/agents/rewrite/tests/` และ test harness ใน `__main__` ของ classification agent)

---

## กิตติกรรมประกาศ

- **สาขาวิชาฟิสิกส์ มหาวิทยาลัยเทคโนโลยีสุรนารี** — เจ้าของโครงการ
- **สถาบันวิจัยดาราศาสตร์แห่งชาติ (NARIT)** — องค์ความรู้เชิงโดเมนและข้อมูล XMM-Newton
- พัฒนาด้วย FastAPI, LangChain, Groq, SciPy และ PostgreSQL

---

## สัญญาอนุญาต

เผยแพร่ภายใต้ [สัญญาอนุญาต MIT](LICENSE)

---

## ผู้พัฒนา

**ปริญญา ดวงกลาง** — AI Engineer / Research Assistant
[GitHub](https://github.com/parinyad123) · parinya.dg@gmail.com
