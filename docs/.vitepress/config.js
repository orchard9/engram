import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Engram',
  description: 'Cognitive graph database for AI memory systems',

  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Getting Started', link: '/getting-started' },
      { text: 'API', link: '/api/' },
      {
        text: 'Spreading Activation',
        items: [
          { text: 'Tutorial', link: '/tutorials/spreading_getting_started' },
          { text: 'How-To', link: '/howto/spreading_performance' },
          { text: 'Monitoring', link: '/howto/spreading_monitoring' },
          { text: 'Debugging', link: '/howto/spreading_debugging' },
          { text: 'Explanation', link: '/explanation/cognitive_spreading' },
          { text: 'Reference', link: '/reference/spreading_api' }
        ]
      }
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
        text: 'Spreading Activation',
        items: [
          { text: 'Tutorial', link: '/tutorials/spreading_getting_started' },
          { text: 'Performance', link: '/howto/spreading_performance' },
          { text: 'Monitoring', link: '/howto/spreading_monitoring' },
          { text: 'Debugging', link: '/howto/spreading_debugging' },
          { text: 'Cognitive Explanation', link: '/explanation/cognitive_spreading' },
          { text: 'API Reference', link: '/reference/spreading_api' }
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
