import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Engram',
  description: 'Cognitive graph database for AI memory systems',

  // Exclude internal documentation from build
  srcExclude: ['internal/**'],

  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Guide', link: '/guide/why-engram' },
      { text: 'Operations', link: '/operations/' },
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

    sidebar: {
      // Guide section sidebar
      '/guide/': [
        {
          text: 'Guide',
          items: [
            { text: 'Why Engram?', link: '/guide/why-engram' },
            { text: 'Core Concepts', link: '/guide/core-concepts' },
            { text: 'When to Use', link: '/guide/when-to-use' },
            { text: 'Architecture Overview', link: '/guide/architecture-overview' }
          ]
        },
        {
          text: 'Get Started',
          items: [
            { text: 'Quick Start', link: '/getting-started' }
          ]
        }
      ],

      // Operations section sidebar
      '/operations/': [
        {
          text: 'Getting Started',
          items: [
            { text: 'Overview', link: '/operations/' },
            { text: 'Production Deployment', link: '/operations/production-deployment' },
            { text: 'Monitoring', link: '/operations/monitoring' },
            { text: 'Backup & Restore', link: '/operations/backup-restore' }
          ]
        },
        {
          text: 'Day-to-Day Operations',
          items: [
            { text: 'Performance Tuning', link: '/operations/performance-tuning' },
            { text: 'Troubleshooting', link: '/operations/troubleshooting' },
            { text: 'Scaling', link: '/operations/scaling' }
          ]
        },
        {
          text: 'Operational Runbooks',
          items: [
            { text: 'Operations Runbook', link: '/operations/runbook' },
            { text: 'Memory Space Migration', link: '/operations/memory-space-migration' },
            { text: 'Metrics Streaming', link: '/operations/metrics-streaming' },
            { text: 'SSE Streaming Health', link: '/operations/sse-streaming-health' }
          ]
        },
        {
          text: 'Feature-Specific Operations',
          items: [
            { text: 'Spreading Activation', link: '/operations/spreading' },
            { text: 'Spreading Validation', link: '/operations/spreading-validation' },
            { text: 'Consolidation Dashboard', link: '/operations/consolidation-dashboard' },
            { text: 'Consolidation Observability', link: '/operations/consolidation-observability' }
          ]
        }
      ],

      // Spreading Activation section sidebar
      '/tutorials/': [
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
        }
      ],
      '/howto/': [
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
        }
      ],
      '/explanation/': [
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
        }
      ],

      // API Reference section sidebar
      '/api/': [
        {
          text: 'API Reference',
          items: [
            { text: 'Overview', link: '/api/' },
            { text: 'Memory Operations', link: '/api/memory' },
            { text: 'System Health', link: '/api/system' }
          ]
        }
      ],

      // Reference section sidebar
      '/reference/': [
        {
          text: 'Reference',
          items: [
            { text: 'Spreading API', link: '/reference/spreading_api' },
            { text: 'CLI Reference', link: '/reference/cli' }
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

      // Default sidebar for other pages
      '/': [
        {
          text: 'Introduction',
          items: [
            { text: 'What is Engram?', link: '/' },
            { text: 'Getting Started', link: '/getting-started' }
          ]
        },
        {
          text: 'Guide',
          items: [
            { text: 'Why Engram?', link: '/guide/why-engram' },
            { text: 'Core Concepts', link: '/guide/core-concepts' }
          ]
        },
        {
          text: 'Operations',
          items: [
            { text: 'Overview', link: '/operations/' },
            { text: 'Production Deployment', link: '/operations/production-deployment' },
            { text: 'Monitoring', link: '/operations/monitoring' }
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
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/orchard9/engram' }
    ]
  }
})
