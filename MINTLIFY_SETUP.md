# Mintlify Documentation Setup

TACT now uses **Mintlify** for beautiful, modern documentation with AI-powered search!

## 🎉 What You Get

- ✅ **Beautiful Modern UI** - Professional, clean design
- ✅ **AI-Powered Search** - Smart search that understands context
- ✅ **Syntax Highlighting** - Beautiful code blocks
- ✅ **Fast Performance** - Lightning-fast page loads
- ✅ **Mobile Responsive** - Works perfectly on all devices
- ✅ **Dark Mode** - Built-in dark/light theme toggle
- ✅ **Analytics** - Track documentation usage
- ✅ **Feedback System** - Users can rate pages and suggest edits

---

## 🚀 Quick Start

### View Documentation Locally

```bash
# Make sure you have Node.js installed
node --version  # Should be v16 or higher

# Install Mintlify (if not already installed)
npm install -g mintlify

# Start the dev server
mintlify dev

# Opens at http://localhost:3000
```

The server will automatically reload when you edit documentation files!

---

## 📁 Documentation Structure

```
TACT/
├── mint.json                 # Mintlify configuration
├── docs/                     # All documentation files
│   ├── index.md             # Home page
│   ├── installation-guide.md
│   ├── getting-started.md
│   ├── data-import-guide.md
│   ├── quickstart.md
│   ├── add-custom-model.md
│   ├── contributing.md
│   ├── api/                 # API documentation
│   │   ├── core/
│   │   ├── adjustments/
│   │   └── utils/
│   └── assets/              # Images, logos, etc.
└── README.md                # Still used as homepage alternate
```

---

## ⚙️ Configuration (`mint.json`)

The `mint.json` file controls all Mintlify settings:

### Key Settings

```json
{
  "name": "TACT",
  "logo": {
    "dark": "/docs/assets/cfars_logo_transparent.png",
    "light": "/docs/assets/cfars_logo_transparent.png"
  },
  "colors": {
    "primary": "#0D9373",    // Your brand color
    "light": "#07C983",
    "dark": "#0D9373"
  },
  "navigation": [...],        // Sidebar navigation
  "search": {
    "prompt": "Search TACT documentation..."
  }
}
```

### Adding New Pages

1. Create your `.md` file in `docs/`
2. Add it to `navigation` in `mint.json`:

```json
{
  "navigation": [
    {
      "group": "Your Section",
      "pages": [
        "docs/your-new-page"  // .md extension omitted
      ]
    }
  ]
}
```

---

## 📝 Writing Documentation

### MDX Support

Mintlify supports MDX (Markdown + JSX), which means you can use:

**Standard Markdown:**
```markdown
# Heading
**Bold** and *italic*
- Lists
- [Links](url)
```

**Code Blocks with Syntax Highlighting:**
````markdown
```python
from tact import TACT
tact = TACT()
```
````

**Callouts/Admonitions:**
```markdown
<Note>
This is an important note!
</Note>

<Warning>
Be careful with this setting
</Warning>

<Tip>
Pro tip: Use this feature!
</Tip>
```

**Tabs:**
```markdown
<Tabs>
  <Tab title="Python">
    ```python
    print("Hello")
    ```
  </Tab>
  <Tab title="Bash">
    ```bash
    echo "Hello"
    ```
  </Tab>
</Tabs>
```

**Accordions:**
```markdown
<AccordionGroup>
  <Accordion title="Question 1">
    Answer to question 1
  </Accordion>
  <Accordion title="Question 2">
    Answer to question 2
  </Accordion>
</AccordionGroup>
```

**Cards:**
```markdown
<CardGroup cols={2}>
  <Card title="Option 1" icon="rocket">
    Description of option 1
  </Card>
  <Card title="Option 2" icon="star">
    Description of option 2
  </Card>
</CardGroup>
```

---

## 🎨 Customization

### Colors

Edit `mint.json`:
```json
{
  "colors": {
    "primary": "#0D9373",
    "light": "#07C983",
    "dark": "#0D9373"
  }
}
```

### Logo

Replace logo files in `docs/assets/` and update `mint.json`:
```json
{
  "logo": {
    "dark": "/docs/assets/your-dark-logo.png",
    "light": "/docs/assets/your-light-logo.png"
  }
}
```

### Favicon

```json
{
  "favicon": "/docs/assets/favicon.ico"
}
```

---

## 🚢 Deployment Options

### Option 1: Mintlify Hosting (Recommended) ⭐

**Easiest option - free for open source!**

1. **Sign up:** https://mintlify.com
2. **Connect GitHub:**
   - Link your TACT repository
   - Mintlify auto-detects `mint.json`
3. **Deploy:**
   - Auto-deploys on every push to main
   - Custom domain support
   - Free SSL certificate
   - CDN included

**URL:** `https://tact.mintlify.app` (or custom domain)

---

### Option 2: Vercel/Netlify (Self-Hosted)

Mintlify can also be deployed to Vercel or Netlify:

**Vercel:**
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel
```

**Netlify:**
```bash
# Install Netlify CLI
npm i -g netlify-cli

# Deploy
netlify deploy
```

---

### Option 3: Static Export

Generate static HTML for any web host:

```bash
mintlify build

# Output in .mintlify/ directory
# Upload to any static host (S3, GitHub Pages, etc.)
```

---

## 🔍 AI Search Configuration

AI search is **automatically enabled** in Mintlify! No extra configuration needed.

### How It Works

- Automatically indexes all your documentation
- Understands natural language queries
- Provides context-aware results
- Learns from user behavior

### Customization

```json
{
  "search": {
    "prompt": "Search TACT documentation...",
    "suggestions": [
      "How do I install TACT?",
      "What is DNV validation?",
      "How to add custom methods?"
    ]
  }
}
```

---

## 📊 Analytics

Enable analytics in `mint.json`:

```json
{
  "analytics": {
    "ga4": {
      "measurementId": "G-XXXXXXXXXX"
    }
  }
}
```

Or use Mintlify's built-in analytics (available in dashboard).

---

## 🔗 Useful Features

### Code Groups

Show code in multiple languages:

````markdown
<CodeGroup>
```python Python
from tact import TACT
tact = TACT()
```

```javascript JavaScript
import TACT from 'tact';
const tact = new TACT();
```
</CodeGroup>
````

### API Endpoints

Document REST APIs:

```markdown
<ApiEndpoint
  method="GET"
  url="/api/v1/adjust"
  description="Run turbulence adjustment"
/>
```

### OpenAPI Integration

Import OpenAPI/Swagger specs:

```json
{
  "openapi": "/path/to/openapi.json"
}
```

---

## 🐛 Troubleshooting

### Issue: "Parsing error"

**Cause:** HTML-like syntax in markdown (e.g., `<500` interpreted as tag)

**Fix:** Use `less than` instead of `<`:
```markdown
❌ You have limited data (<500 points)
✅ You have limited data (less than 500 points)
```

---

### Issue: "Command not found: mintlify"

**Fix:**
```bash
npm install -g mintlify
```

---

### Issue: Images not showing

**Cause:** Incorrect path in markdown

**Fix:** Use absolute paths from root:
```markdown
❌ ![Logo](../assets/logo.png)
✅ ![Logo](/docs/assets/logo.png)
```

---

### Issue: "Port already in use"

**Fix:** Mintlify auto-finds available port:
```bash
mintlify dev
# Will use 3001, 3002, etc. if 3000 is taken
```

Or specify port:
```bash
mintlify dev --port 3005
```

---

## 📚 Resources

- **Mintlify Docs:** https://mintlify.com/docs
- **Component Library:** https://mintlify.com/docs/content/components
- **Examples:** https://mintlify.com/showcase
- **Support:** https://mintlify.com/community

---

## 🎯 Best Practices

### 1. Keep Navigation Clean
- Max 7-8 items per group
- Use logical grouping
- Order by importance

### 2. Use Callouts Effectively
```markdown
<Tip>
For most use cases, SS-SF is the recommended method.
</Tip>
```

### 3. Add Code Copy Buttons
Code blocks automatically get copy buttons!

### 4. Link Related Pages
```markdown
See also: [Installation Guide](installation-guide)
```

### 5. Use Search Keywords
Add metadata to pages:
```markdown
---
title: "Getting Started with TACT"
description: "Complete guide to installing and using TACT"
---
```

---

## 🔄 Migration from MkDocs

The migration is complete! Here's what changed:

| Aspect | MkDocs | Mintlify |
|--------|--------|----------|
| **Config** | `mkdocs.yml` | `mint.json` |
| **Dev Server** | `mkdocs serve` | `mintlify dev` |
| **Build** | `mkdocs build` | `mintlify build` |
| **Hosting** | Manual | Auto-deploy |
| **Search** | Basic | AI-powered |
| **Components** | Limited | Rich library |

All your existing `.md` files work as-is! No content changes needed (except minor HTML fixes).

---

## ✅ Next Steps

1. **View Local Docs:**
   ```bash
   mintlify dev
   ```

2. **Test Navigation:** Click through all pages

3. **Deploy:**
   - Push to GitHub
   - Connect repo to Mintlify
   - Auto-deploys!

4. **Share:**
   - Share `mintlify.app` URL
   - Add to README
   - Tweet about it!

---

## 🎉 You're All Set!

Your documentation is now powered by Mintlify with:
- ✅ Beautiful modern UI
- ✅ AI search
- ✅ Syntax highlighting
- ✅ Dark mode
- ✅ Analytics ready
- ✅ Mobile responsive

Run `mintlify dev` and enjoy! 🚀
