// Page transition system for specific routes
// Only handles: index -> how-it-works, index -> find-missing-pet, index -> register-missing-pet

(function() {
    'use strict';
    
    // Configuration for specific transitions
    const TRANSITION_ROUTES = {
        'how-it-works.php': true,
        'find-missing-pet.php': true,
        'register-missing-pet.php': true
    };
    
    const TRANSITION_DURATION = 800; // Slower, more relaxed timing
    
    // Create overlay element
    function createTransitionOverlay() {
        const overlay = document.createElement('div');
        overlay.id = 'page-transition-overlay';
        overlay.innerHTML = `
            <div class="loading-content">
                <div class="loading-paw">
                    <img src="assets/Logos/pawprint-blue 1.png" alt="Loading">
                </div>
                <p class="loading-text">Loading...</p>
            </div>
        `;
        document.body.appendChild(overlay);
        return overlay;
    }
    
    // Get or create overlay
    function getTransitionOverlay() {
        let overlay = document.getElementById('page-transition-overlay');
        if (!overlay) {
            overlay = createTransitionOverlay();
        }
        return overlay;
    }
    
    // Check if we should use transitions for this link
    function shouldUseTransition(href) {
        if (!href) return false;
        
        // Extract filename from href
        const filename = href.split('/').pop().split('?')[0];
        return TRANSITION_ROUTES[filename] === true;
    }
    
    // Navigate with transition
    function navigateWithTransition(url) {
        try {
            const overlay = getTransitionOverlay();
            
            // Show overlay
            overlay.classList.add('active');
            
            // Navigate after overlay is shown
            setTimeout(() => {
                window.location.href = url;
            }, 600); // More relaxed timing to show transition
        } catch (error) {
            // Fallback to direct navigation if transition fails
            console.warn('Page transition failed, using direct navigation:', error);
            window.location.href = url;
        }
    }
    
    // Initialize page transitions
    function initPageTransitions() {
        // Only run on index.php
        const currentPage = window.location.pathname.split('/').pop() || 'index.php';
        if (currentPage !== 'index.php' && currentPage !== '') return;
        
        // Handle clicks on feature cards (which are anchor tags)
        document.addEventListener('click', function(e) {
            // Find the closest anchor tag
            const link = e.target.closest('a.feature-card');
            if (!link) return;
            
            const href = link.getAttribute('href');
            if (!shouldUseTransition(href)) return;
            
            // Prevent default navigation
            e.preventDefault();
            
            // Add visual feedback using existing CSS classes
            link.classList.add('card-clicked');
            
            // Start transition after brief delay to show card animation
            setTimeout(() => {
                navigateWithTransition(href);
            }, 250);
        });
        
        // Preserve existing card hover effects
        // This ensures compatibility with the existing script.js
    }
    
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initPageTransitions);
    } else {
        initPageTransitions();
    }
    
    // Handle page load completion - single unified handler
    let overlayHandled = false;
    
    function handleOverlayHide() {
        if (overlayHandled) return;
        overlayHandled = true;
        
        const overlay = document.getElementById('page-transition-overlay');
        if (overlay && overlay.classList.contains('active')) {
            // Hide overlay when page is fully loaded
            setTimeout(() => {
                overlay.classList.remove('active');
            }, 400);
        }
    }
    
    // Use only window load event for consistent behavior
    window.addEventListener('load', handleOverlayHide);
    
})();