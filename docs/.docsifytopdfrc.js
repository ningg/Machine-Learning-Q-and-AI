module.exports = {
    // 必需：指定目录文件路径数组
    contents: ["_sidebar.md"],
    
    // 必需：PDF 输出路径
    pathToPublic: "pdf/大模型技术30讲(英文&中文批注)_LLM_30_Essential_Lectures_AI.pdf",
    
    // 可选：PDF 选项（Puppeteer 配置）
    pdfOptions: {
      format: "A4",
      margin: {
        top: "20mm",
        right: "10mm",
        bottom: "20mm",
        left: "18mm"
      },
      printBackground: true,
      displayHeaderFooter: true,
      headerTemplate: '<div style="font-size: 10px; text-align: center; width: 100%; color: gray;"><span class="date"></span>&emsp;&emsp;|&emsp;&emsp;<span class="title"></span>&emsp;&emsp;|&emsp;&emsp;<span style="color: #42b983;">https://ningg.top/Machine-Learning-Q-and-AI/</span></div>',
      footerTemplate: '<div style="font-size: 10px; text-align: center; width: 100%;">第 <span class="pageNumber"></span> 页，共 <span class="totalPages"></span> 页</div>',
      outline: true,
    },
    
    // 必填：Chrome 浏览器可执行文件路径
    chromeExecutablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    
    // 可选：是否删除临时文件
    removeTemp: true,
    
    // 可选：媒体类型模拟（print 或 screen）
    emulateMedia: "screen",
    
    // 可选：静态文件路径
    pathToStatic: "static",
    
    // 可选：主 Markdown 文件名
    mainMdFilename: "README.md",
    
    // 可选：docsify 入口点路径
    pathToDocsifyEntryPoint: ".",
    
    // 可选：分页符配置
    pageBreak: {
      // 是否启用分页符
      enabled: true,
      // 分页符样式：'div' 或 'css'
      type: 'div',
      // 自定义分页符 HTML（当 type 为 'div' 时使用）
      html: '<div style="page-break-after: always;"></div>',
      // 自定义 CSS 样式（当 type 为 'css' 时使用）
      css: `
        .markdown-section h1, .markdown-section h2, .markdown-section h3 {
          page-break-before: always;
        }
        .markdown-section h1:first-child, .markdown-section h2:first-child, .markdown-section h3:first-child {
          page-break-before: avoid;
        }
        .markdown-section {
          page-break-inside: avoid;
        }
        .markdown-section h1, .markdown-section h2, .markdown-section h3 {
          break-before: page;
        }
        .markdown-section h1:first-child, .markdown-section h2:first-child, .markdown-section h3:first-child {
          break-before: avoid;
        }
      `
    }
  };