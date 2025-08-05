"""
Advanced Search Filters
UI components for advanced search filtering and content discovery
"""
import streamlit as st
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd

from analytics.metrics_collector import get_metrics_collector


class AdvancedSearchFilters:
    """Advanced search filters for content discovery"""
    
    def __init__(self):
        self.metrics_collector = get_metrics_collector()
        
        # Initialize filter state
        if 'search_filters' not in st.session_state:
            st.session_state.search_filters = {
                "authors": [],
                "sources": [],
                "min_score": None,
                "keywords": [],
                "exclude_keywords": [],
                "min_text_length": None,
                "max_text_length": None,
                "date_range": None
            }
    
    def render_filters_sidebar(self) -> Dict[str, Any]:
        """Render advanced search filters in sidebar"""
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Advanced Filters")
        
        with st.sidebar.expander("📋 Content Filters", expanded=False):
            filters = self._render_content_filters()
        
        with st.sidebar.expander("👥 Author & Source", expanded=False):
            author_source_filters = self._render_author_source_filters()
            filters.update(author_source_filters)
        
        with st.sidebar.expander("📊 Quality Filters", expanded=False):
            quality_filters = self._render_quality_filters()
            filters.update(quality_filters)
        
        with st.sidebar.expander("🔤 Text Filters", expanded=False):
            text_filters = self._render_text_filters()
            filters.update(text_filters)
        
        # Filter actions
        st.sidebar.markdown("---")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🔄 Reset Filters", key="reset_filters"):
                self._reset_filters()
                st.rerun()
        
        with col2:
            if st.button("💾 Save Filters", key="save_filters"):
                self._save_filters(filters)
        
        # Show active filters summary
        active_count = self._count_active_filters(filters)
        if active_count > 0:
            st.sidebar.info(f"🎯 {active_count} filter(s) active")
        
        return filters
    
    def render_filters_expander(self) -> Dict[str, Any]:
        """Render advanced search filters in main area expander"""
        
        with st.expander("🔍 Advanced Search Filters", expanded=False):
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Content Filters**")
                content_filters = self._render_content_filters()
            
            with col2:
                st.markdown("**Author & Source**")
                author_source_filters = self._render_author_source_filters()
            
            with col3:
                st.markdown("**Quality & Text**")
                quality_filters = self._render_quality_filters()
                text_filters = self._render_text_filters()
            
            # Combine all filters
            filters = {**content_filters, **author_source_filters, **quality_filters, **text_filters}
            
            # Filter actions
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🔄 Reset All", key="reset_filters_main"):
                    self._reset_filters()
                    st.rerun()
            
            with col2:
                if st.button("💾 Save Preset", key="save_filters_main"):
                    self._save_filters(filters)
            
            with col3:
                if st.button("📊 Filter Stats", key="filter_stats"):
                    self._show_filter_statistics(filters)
            
            with col4:
                active_count = self._count_active_filters(filters)
                st.metric("Active Filters", active_count)
        
        return filters
    
    def _render_content_filters(self) -> Dict[str, Any]:
        """Render content-based filters"""
        
        filters = {}
        
        # Keywords inclusion
        keywords_input = st.text_input(
            "🔍 Include Keywords",
            value=", ".join(st.session_state.search_filters.get("keywords", [])),
            help="Comma-separated keywords that must be present",
            key="include_keywords"
        )
        
        if keywords_input.strip():
            filters["keywords"] = [k.strip() for k in keywords_input.split(",") if k.strip()]
        
        # Keywords exclusion
        exclude_keywords_input = st.text_input(
            "🚫 Exclude Keywords",
            value=", ".join(st.session_state.search_filters.get("exclude_keywords", [])),
            help="Comma-separated keywords to exclude",
            key="exclude_keywords"
        )
        
        if exclude_keywords_input.strip():
            filters["exclude_keywords"] = [k.strip() for k in exclude_keywords_input.split(",") if k.strip()]
        
        return filters
    
    def _render_author_source_filters(self) -> Dict[str, Any]:
        """Render author and source filters"""
        
        filters = {}
        
        # Get available authors and sources from analytics
        content_analytics = self.metrics_collector.get_content_analytics()
        author_stats = content_analytics.get("author_stats", {})
        
        search_analytics = self.metrics_collector.get_search_analytics(hours_back=168)  # Last week
        source_dist = search_analytics.get("source_distribution", {})
        
        # Author filter
        if author_stats:
            available_authors = sorted(author_stats.keys())
            selected_authors = st.multiselect(
                "👥 Select Authors",
                options=available_authors,
                default=st.session_state.search_filters.get("authors", []),
                help="Filter by specific authors",
                key="author_filter"
            )
            
            if selected_authors:
                filters["authors"] = selected_authors
        else:
            st.info("No author data available")
        
        # Source filter
        if source_dist:
            available_sources = sorted(source_dist.keys())
            selected_sources = st.multiselect(
                "📂 Select Sources",
                options=available_sources,
                default=st.session_state.search_filters.get("sources", []),
                help="Filter by content sources",
                key="source_filter"
            )
            
            if selected_sources:
                filters["sources"] = selected_sources
        else:
            st.info("No source data available")
        
        return filters
    
    def _render_quality_filters(self) -> Dict[str, Any]:
        """Render quality-based filters"""
        
        filters = {}
        
        # Minimum similarity score
        min_score = st.slider(
            "📊 Minimum Similarity Score",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.search_filters.get("min_score", 0.0),
            step=0.05,
            help="Filter results by minimum similarity score",
            key="min_score_filter"
        )
        
        if min_score > 0.0:
            filters["min_score"] = min_score
        
        return filters
    
    def _render_text_filters(self) -> Dict[str, Any]:
        """Render text length filters"""
        
        filters = {}
        
        # Text length range
        st.markdown("📏 **Text Length Filter**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            min_length = st.number_input(
                "Min Characters",
                min_value=0,
                max_value=10000,
                value=st.session_state.search_filters.get("min_text_length", 0),
                step=50,
                key="min_text_length"
            )
            
            if min_length > 0:
                filters["min_text_length"] = min_length
        
        with col2:
            max_length = st.number_input(
                "Max Characters", 
                min_value=0,
                max_value=10000,
                value=st.session_state.search_filters.get("max_text_length", 0),
                step=50,
                key="max_text_length"
            )
            
            if max_length > 0:
                filters["max_text_length"] = max_length
        
        return filters
    
    def _reset_filters(self):
        """Reset all filters to default values"""
        st.session_state.search_filters = {
            "authors": [],
            "sources": [],
            "min_score": None,
            "keywords": [],
            "exclude_keywords": [],
            "min_text_length": None,
            "max_text_length": None,
            "date_range": None
        }
        
        # Clear filter widget states
        filter_keys = [
            "include_keywords", "exclude_keywords", "author_filter", "source_filter",
            "min_score_filter", "min_text_length", "max_text_length"
        ]
        
        for key in filter_keys:
            if key in st.session_state:
                del st.session_state[key]
    
    def _save_filters(self, filters: Dict[str, Any]):
        """Save current filter configuration"""
        st.session_state.search_filters = filters
        st.success("🎯 Filters saved for this session!")
    
    def _count_active_filters(self, filters: Dict[str, Any]) -> int:
        """Count number of active filters"""
        active_count = 0
        
        for key, value in filters.items():
            if key in ["keywords", "exclude_keywords", "authors", "sources"]:
                if value and len(value) > 0:
                    active_count += 1
            elif value is not None and value != 0:
                active_count += 1
        
        return active_count
    
    def _show_filter_statistics(self, filters: Dict[str, Any]):
        """Show statistics about current filters"""
        
        st.markdown("### 📊 Filter Statistics")
        
        if not filters:
            st.info("No filters currently active")
            return
        
        # Show active filters
        for filter_type, filter_value in filters.items():
            if filter_value:
                if isinstance(filter_value, list):
                    st.write(f"**{filter_type}:** {', '.join(map(str, filter_value))}")
                else:
                    st.write(f"**{filter_type}:** {filter_value}")
    
    def get_filter_suggestions(self) -> Dict[str, List[str]]:
        """Get filter suggestions based on analytics"""
        
        suggestions = {
            "popular_keywords": [],
            "trending_authors": [],
            "active_sources": []
        }
        
        # Get analytics data
        search_analytics = self.metrics_collector.get_search_analytics(hours_back=168)
        content_analytics = self.metrics_collector.get_content_analytics()
        
        # Popular query terms (extract keywords)
        top_queries = search_analytics.get("top_queries", [])
        all_terms = []
        for query, count in top_queries:
            terms = query.lower().split()
            all_terms.extend(terms)
        
        # Count term frequency
        from collections import Counter
        term_counts = Counter(all_terms)
        suggestions["popular_keywords"] = [term for term, count in term_counts.most_common(10)]
        
        # Trending authors
        top_authors = search_analytics.get("top_authors", [])
        suggestions["trending_authors"] = [author for author, count in top_authors[:10]]
        
        # Active sources
        source_dist = search_analytics.get("source_distribution", {})
        suggestions["active_sources"] = list(source_dist.keys())
        
        return suggestions
    
    def render_quick_filters(self) -> Dict[str, Any]:
        """Render quick filter buttons"""
        
        st.markdown("**🚀 Quick Filters**")
        
        suggestions = self.get_filter_suggestions()
        filters = {}
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("*Popular Keywords*")
            for keyword in suggestions["popular_keywords"][:5]:
                if st.button(f"#{keyword}", key=f"quick_keyword_{keyword}"):
                    filters["keywords"] = [keyword]
        
        with col2:
            st.markdown("*Trending Authors*")
            for author in suggestions["trending_authors"][:5]:
                if st.button(f"@{author}", key=f"quick_author_{author}"):
                    filters["authors"] = [author]
        
        with col3:
            st.markdown("*Active Sources*")
            for source in suggestions["active_sources"][:5]:
                if st.button(f"📂{source}", key=f"quick_source_{source}"):
                    filters["sources"] = [source]
        
        return filters
    
    def apply_filters_to_query(self, base_query: str, filters: Dict[str, Any]) -> str:
        """Enhance base query with filter information"""
        
        enhanced_query = base_query
        
        # Add keyword filters to query context
        if filters.get("keywords"):
            keyword_context = " ".join(filters["keywords"])
            enhanced_query = f"{base_query} {keyword_context}"
        
        return enhanced_query
    
    def get_filter_summary(self, filters: Dict[str, Any]) -> str:
        """Get human-readable filter summary"""
        
        if not filters:
            return "No filters applied"
        
        summary_parts = []
        
        if filters.get("authors"):
            authors_str = ", ".join(filters["authors"][:3])
            if len(filters["authors"]) > 3:
                authors_str += f" and {len(filters['authors']) - 3} others"
            summary_parts.append(f"Authors: {authors_str}")
        
        if filters.get("sources"):
            sources_str = ", ".join(filters["sources"][:2])
            if len(filters["sources"]) > 2:
                sources_str += f" and {len(filters['sources']) - 2} others"
            summary_parts.append(f"Sources: {sources_str}")
        
        if filters.get("keywords"):
            keywords_str = ", ".join(filters["keywords"][:3])
            summary_parts.append(f"Keywords: {keywords_str}")
        
        if filters.get("min_score"):
            summary_parts.append(f"Min Score: {filters['min_score']:.2f}")
        
        if filters.get("min_text_length"):
            summary_parts.append(f"Min Length: {filters['min_text_length']} chars")
        
        return " | ".join(summary_parts) if summary_parts else "No filters applied"


# Global instance
_search_filters = None

def get_search_filters() -> AdvancedSearchFilters:
    """Get global search filters instance"""
    global _search_filters
    if _search_filters is None:
        _search_filters = AdvancedSearchFilters()
    return _search_filters