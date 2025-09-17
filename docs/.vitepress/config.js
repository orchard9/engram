import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Engram',
  description: 'Cognitive graph database for AI memory systems',

  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Getting Started', link: '/getting-started' },
      { text: 'API', link: '/api/' }
    ],

    sidebar: [
      {
        text: 'Introduction',
        items: [
          { text: 'What is Engram?', link: '/' },
          { text: 'Getting Started', link: '/getting-started' }
        ]
      },
      {
        text: 'API Reference',
        items: [
          { text: 'Overview', link: '/api/' },
          { text: 'Memory Operations', link: '/api/memory' },
          { text: 'System Health', link: '/api/system' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/orchard9/engram' }
    ]
  }
})