class PageHeader extends HTMLElement {
    connectedCallback() {
        const currentPath = window.location.pathname;
        this.innerHTML = `
        <header class="page-header">
            <nav>
                <a href="/" class="${currentPath == '/' ? 'active' : ''}">Home</a>
            </nav>
        </header>
        `;
    }
}

class PageFooter extends HTMLElement {
    connectedCallback() {
        const year = new Date().getFullYear();
        this.innerHTML = `
        <footer class="page-footer">
            &copy; ${year} Reddit Sentiment Dashboard
        </footer>
        `;
    }
}

customElements.define('page-header', PageHeader);
customElements.define('page-footer', PageFooter);