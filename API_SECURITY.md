# 🔒 API Security Setup

## Protecting Your Google API Credentials

Your API keys are now **safely excluded from Git** using `.gitignore`.

### Quick Setup

1. **Copy the example file:**
   ```bash
   copy .env.example .env
   ```

2. **Edit `.env` with your credentials:**
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   GOOGLE_SEARCH_ENGINE_ID=your_actual_engine_id_here
   ```

3. **Done!** The `.env` file is automatically loaded and **never** pushed to GitHub.

---

## Getting Your API Credentials

### Google API Key
1. Go to [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
2. Create a new project (or select existing)
3. Enable "Custom Search API"
4. Create credentials → API Key
5. Copy the API key to `.env`

### Search Engine ID
1. Go to [Google Custom Search](https://cse.google.com/cse/)
2. Create a new search engine
3. Configure to search the entire web
4. Copy the Search Engine ID to `.env`

---

## Files Overview

| File | Purpose | Git Status |
|------|---------|------------|
| `.env` | Your actual credentials | ❌ **NOT** tracked (in `.gitignore`) |
| `.env.example` | Template with placeholders | ✅ Safe to commit |
| `.gitignore` | Protects sensitive files | ✅ Safe to commit |

---

## Verification

Check that your `.env` is protected:

```bash
git status
```

You should **NOT** see `.env` in the list. If you do, run:

```bash
git rm --cached .env
git commit -m "Remove .env from tracking"
```

---

## Fallback Mode

The system works **without API keys** using built-in product database!

- With API: Real-time Google search verification
- Without API: Local fallback database (still accurate)

---

## ⚠️ Important Notes

1. **Never commit `.env`** - It contains your private API keys
2. **Always use `.env.example`** for templates
3. **Share `.env.example`** - Safe for others to copy
4. **Keep `.env` local** - Each developer uses their own credentials

---

## Emergency: If You Accidentally Committed API Keys

1. **Regenerate your API key immediately** at Google Cloud Console
2. **Remove from Git history:**
   ```bash
   git filter-branch --force --index-filter "git rm --cached --ignore-unmatch .env" --prune-empty --tag-name-filter cat -- --all
   git push origin --force --all
   ```
3. **Update `.env` with new key**

---

## Questions?

- The app loads `.env` automatically on startup
- No code changes needed - just set up `.env` once
- Check `utils/google_search.py` to see how it loads credentials
