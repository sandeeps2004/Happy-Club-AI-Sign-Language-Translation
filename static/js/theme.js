/**
 * Theme toggle — simple, no lag.
 * FOUC prevention is handled by inline <script> in <head>.
 */
function toggleDarkMode() {
    var isDark = document.documentElement.classList.toggle('dark');
    localStorage.setItem('darkMode', isDark);
}
