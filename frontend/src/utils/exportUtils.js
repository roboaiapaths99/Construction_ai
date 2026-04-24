// CSV Export utility
export const exportToCSV = (data, filename) => {
  if (!data || data.length === 0) {
    console.warn('No data to export');
    return;
  }

  const headers = Object.keys(data[0]);
  const csvContent = [
    headers.map(h => `"${h}"`).join(','),
    ...data.map(row =>
      headers
        .map(header => {
          const value = row[header];
          // Handle special characters and newlines in CSV
          if (value === null || value === undefined) {
            return '""';
          }
          const stringValue = String(value);
          return `"${stringValue.replace(/"/g, '""')}"`;
        })
        .join(',')
    ),
  ].join('\n');

  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);

  link.setAttribute('href', url);
  link.setAttribute('download', `${filename}-${new Date().toISOString().split('T')[0]}.csv`);
  link.style.visibility = 'hidden';

  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

// JSON Export utility
export const exportToJSON = (data, filename) => {
  if (!data) {
    console.warn('No data to export');
    return;
  }

  const jsonContent = JSON.stringify(data, null, 2);
  const blob = new Blob([jsonContent], { type: 'application/json;charset=utf-8;' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);

  link.setAttribute('href', url);
  link.setAttribute('download', `${filename}-${new Date().toISOString().split('T')[0]}.json`);
  link.style.visibility = 'hidden';

  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

// PDF Export utility (requires jsPDF - needs to be added to package.json)
export const exportToPDF = (data, filename, options = {}) => {
  try {
    // This requires jsPDF to be installed: npm install jspdf
    const { jsPDF } = window.jspdf || {};
    if (!jsPDF) {
      console.warn('jsPDF not available. Please install jspdf package.');
      return;
    }

    const doc = new jsPDF(options.orientation || 'p', 'mm', 'a4');
    const pageHeight = doc.internal.pageSize.getHeight();
    const pageWidth = doc.internal.pageSize.getWidth();
    const margin = 10;
    let currentY = margin;

    // Add title
    doc.setFontSize(16);
    doc.text(filename, margin, currentY);
    currentY += 10;

    // Add timestamp
    doc.setFontSize(10);
    doc.text(`Generated: ${new Date().toLocaleString()}`, margin, currentY);
    currentY += 10;

    // Add data as table
    if (Array.isArray(data) && data.length > 0) {
      const headers = Object.keys(data[0]);
      const rows = data.map(item => headers.map(h => String(item[h] || '')));

      doc.autoTable({
        head: [headers],
        body: rows,
        startY: currentY,
        margin: margin,
        didDrawPage: function() {
          const pageCount = doc.internal.getPages().length;
          doc.setFontSize(8);
          doc.text(
            `Page ${pageCount}`,
            pageWidth / 2,
            pageHeight - 5,
            { align: 'center' }
          );
        }
      });
    }

    doc.save(`${filename}-${new Date().toISOString().split('T')[0]}.pdf`);
  } catch (error) {
    console.error('Error exporting to PDF:', error);
  }
};

// Generate Report utility
export const generateReport = (title, sections) => {
  let htmlContent = `
    <!DOCTYPE html>
    <html>
    <head>
      <title>${title}</title>
      <style>
        body {
          font-family: Arial, sans-serif;
          margin: 20px;
          color: #333;
        }
        h1 { color: #1f2937; border-bottom: 3px solid #3b82f6; padding-bottom: 10px; }
        h2 { color: #374151; margin-top: 30px; }
        table {
          width: 100%;
          border-collapse: collapse;
          margin: 20px 0;
        }
        th, td {
          border: 1px solid #e5e7eb;
          padding: 12px;
          text-align: left;
        }
        th {
          background-color: #f3f4f6;
          font-weight: bold;
        }
        .timestamp {
          color: #6b7280;
          font-size: 12px;
          margin-top: 20px;
        }
        .section {
          margin: 20px 0;
          padding: 15px;
          background-color: #f9fafb;
          border-left: 4px solid #3b82f6;
        }
      </style>
    </head>
    <body>
      <h1>${title}</h1>
  `;

  sections.forEach(section => {
    htmlContent += `<div class="section">`;
    if (section.title) {
      htmlContent += `<h2>${section.title}</h2>`;
    }
    if (section.content) {
      if (typeof section.content === 'string') {
        htmlContent += `<p>${section.content}</p>`;
      } else if (Array.isArray(section.content)) {
        htmlContent += `
          <table>
            <thead>
              <tr>
                ${Object.keys(section.content[0])
                  .map(key => `<th>${key}</th>`)
                  .join('')}
              </tr>
            </thead>
            <tbody>
              ${section.content
                .map(
                  row => `
                <tr>
                  ${Object.values(row)
                    .map(value => `<td>${value}</td>`)
                    .join('')}
                </tr>
              `
                )
                .join('')}
            </tbody>
          </table>
        `;
      }
    }
    htmlContent += `</div>`;
  });

  htmlContent += `
    <div class="timestamp">
      Generated on ${new Date().toLocaleString()}
    </div>
    </body>
    </html>
  `;

  const printWindow = window.open('', '', 'width=1000,height=600');
  printWindow.document.write(htmlContent);
  printWindow.document.close();
  printWindow.print();
};
