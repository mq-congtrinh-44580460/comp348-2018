<?xml version="1.0" encoding="UTF-8"?>

<xsl:stylesheet version="1.0"
                xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

  <xsl:output method="html"
              doctype-system="about:legacy-compat"
              encoding="UTF-8"
              indent="yes" />

  <xsl:template match="/">
    <html>
      <head>
        <title>List of Students</title>

        <link rel="stylesheet" href="students.css" />
      </head>
      <body>
        <table class="students">
          <thead>
            <tr>
              <th>Student Number</th>
              <th>Last Name</th>
              <th>First Name</th>
            </tr>
          </thead>
          <tbody>
            <xsl:for-each select="students/student">
            <xsl:sort select="@num"/>
            <tr>
              <td><xsl:value-of select="@num"/></td>
              <td><xsl:value-of select="last_name"/></td>
              <td><xsl:value-of select="first_name"/></td>
            </tr>
            </xsl:for-each>
          </tbody>
        </table>
      </body>
    </html>
  </xsl:template>

</xsl:stylesheet>